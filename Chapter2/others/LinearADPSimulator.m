classdef LinearADPSimulator < handle
	
	properties
		
% 		x_save=[];
% 		t_save=[];
		
		% System matrices used for simulation purpose
		A = [-0.4125    -0.0248        0.0741     0.0089   0          0;
			101.5873    -7.2651        2.7608     2.8068   0          0;
			0.0704       0.0085       -0.0741    -0.0089   0          0.0200;
			0.0878       0.2672        0         -0.3674   0.0044     0.3962;
			-1.8414      0.0990        0          0       -0.0343    -0.0330;
			0            0             0       -359      187.5364   -87.0316];
		
		B = [-0.0042  0.0064
			-1.0360  1.5849
			0.0042  0;
			0.1261  0;
			0      -0.0168;
			0       0];
		
        xn = 6;
        un = 2;%
		%[xn,un] = size(B);%size of B. un:column #, xn:row #
		
		% Set the weighting matrices for the cost function
		Q = diag([1 1 0.1 0.1 0.1 0.1]);
		R = eye(2);
		
		% Initialize the feedback gain matrix
		K  = zeros(un,xn);  %Only if A is Hurwitz, K can be set as zero.
		N  = 100;           %Length of the window, should be at least xn^2+2*xn*un
		MaxIteration = 10;  %Max iteration times
		T  = 0.01;          %Length of each integration interval
		
		x0=[10;2;10;2;-1;-2]; %Initial condition
		
		expl_noise_freq = (rand(un,100)-.5)*1000; % Exploration noise frequencies
		
		% Matrices to collect online data and perform learning
		Dxx=[];
		XX=[];
		XU=[];
		
		% Initial condition of the augmented system
		X=[x0;kron(x0',x0')';kron(x0,zeros(un,1))]';
		
		% Run the simulation and obtain the data matrices \delta_{xx}, I_{xx},
		% and I_{xu}
		
		T_pl = Inf;     % Time point of entering the post-learning period
		
		
		P_old = zeros(xn);
		P = eye(xn)*10;    % Initialize the previous cost matrix
		it = 0;            % Counter for iterations
		p_save = [];       % Track the cost matrices in all the iterations
		k_save = [];       % Track the feedback gain matrix in each iterations
		
		%[K0,P0] = lqr(A,B,Q,R) % Calculate the ideal solution for comparion purpose
		%k_save  = norm(K-K0);  % keep track of the differences between the actual K
	end
	
	methods
		%function obj = LinearADPSimulator()
		%	%bla
		%	obj = 1;
		%end
		
		function collectOnlineData(obj)
			
			for i=1:obj.N
				% Simulation the system and at the same time collect online info.
				[t,X] = ode45(@mysys,[(i-1)*obj.T,i*obj.T],X(end,:));
				
				%Append new data to the data matrices
				Dxx=[Dxx;kron(X(end,1:xn),X(end,1:xn))-kron(X(1,1:xn),X(1,1:xn))];
				XX=[XX;X(end,xn+1:xn+xn^2)-X(1,xn+1:xn+xn^2)];
				XU=[XU;X(end,xn+xn^2+1:end)-X(1,xn+xn^2+1:end)];
				
				% Keep track of the system trajectories
				obj.x_save=[x_save;X];
				obj.t_save=[t_save;t];
			end
		end
		
		function offPolicyLearning(obj)
			
			while norm(obj.P-obj.P_old)>1e-8 & it < obj.MaxIteration
				it = it+1;                        % Update and display the # of iters
				obj.P_old = obj.P;                        % Update the previous cost matrix
				QK = obj.Q+ obj.K'*obj.R*obj.K;               % Update the Qk matrix
				X2 = obj.XX*kron(eye(obj.xn),obj.K');         %
				X1=[obj.Dxx,-X2-obj.XU];                % Left-hand side of the key equation
				Y = -obj.XX*QK(:);                    % Right-hand side of the key equation
				pp = pinv(X1)*Y;                % Solve the equations in the LS sense
				P = reshape(pp(1:obj.xn*obj.xn), [obj.xn, obj.xn]);  % Reconstruct the symmetric matrix
				P = (P + P')/2;
				obj.p_save = [obj.p_save,norm(obj.P-obj.P0)];   % Keep track of the cost matrix
				BPv = pp(end-(obj.xn*obj.un-1):end);
				obj.K = inv(obj.R)*reshape(BPv,obj.un,obj.xn)/2;% Get the improved gain matrix
				disp(['K_', num2str(it), '=']);
				disp(K);
				obj.k_save = [obj.k_save,norm(obj.K-obj.K0)];     % Keep track of the control gains
			end
		end
		
		function postLearningSimulation(obj)
			% Post-learning simulation
			[tt,xx] = ode45(@mysys,[t(end) 50],obj.X(end,:)');
			
			% Keep track of the post-learning trajectories
			t_final=[t_save;tt];
			x_final=[x_save;xx];
		end
		% The following nested function gives the dynamics of the sytem. Also,
		% integraters are included for the purpose of data collection.
	end
end

% ------------------------ Utilities ------------------------------------ %
function dX = mysys(t,X,obj)

x = X(1:obj.xn);

if obj.learningStatus == 0;  % 0: unlearned; 1: learned
	u = sum(sin(expl_noise_freq*t),2)/100;
else
	u = -obj.K*x; % Exploitation
end

dx  = obj.A*x + obj.B*u;
dxx = kron(x',x')';
dux = kron(x',u')';
dX  = [dx;dxx;dux];
end
