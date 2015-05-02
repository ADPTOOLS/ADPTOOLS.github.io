function []= LinearADPSimulator()

%%  initialization
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

[xn,un] = size(B);%size of B. un:column #, xn:row #

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

ifLearned = 0;

x_save=[];
t_save=[];

for i=1:N
	% Simulation the system and at the same time collect online info.
	[t,X] = ode45(@mysys, [(i-1)*T,i*T],X(end,:));
	
	%Append new data to the data matrices
	Dxx=[Dxx;kron(X(end,1:xn),X(end,1:xn))-kron(X(1,1:xn),X(1,1:xn))];
	XX=[XX;X(end,xn+1:xn+xn^2)-X(1,xn+1:xn+xn^2)];
	XU=[XU;X(end,xn+xn^2+1:end)-X(1,xn+xn^2+1:end)];
	
	% Keep track of the system trajectories
	x_save=[x_save;X];
	t_save=[t_save;t];
end

ifLearned = 1;

P_old = zeros(xn);
P = eye(xn)*10;    % Initialize the previous cost matrix
it = 0;            % Counter for iterations
p_save = [];       % Track the cost matrices in all the iterations
k_save = [];       % Track the feedback gain matrix in each iterations

[K0,P0] = lqr(A,B,Q,R) % Calculate the ideal solution for comparion purpose
k_save  = norm(K-K0);  % keep track of the differences between the actual K
% and the idea valu

% Off policy iteration
while norm(P-P_old)>1e-8 & it<MaxIteration
	it=it+1;                        % Update and display the # of iters
	P_old=P;                        % Update the previous cost matrix
	QK=Q+K'*R*K;                    % Update the Qk matrix
         
	Theta=[Dxx,-XX*kron(eye(xn),K')-XU];                % Left-hand side of the key equation
	Xi=-XX*QK(:);                    % Right-hand side of the key equation
	pp = pinv(Theta)*Xi;                % Solve the equations in the LS sense
	P = reshape(pp(1:xn*xn), [xn, xn]);  % Reconstruct the symmetric matrix
	P = (P + P')/2;
	p_save = [p_save,norm(P-P0)];   % Keep track of the cost matrix
	BPv = pp(end-(xn*un-1):end);
	K = inv(R)*reshape(BPv,un,xn)/2;% Get the improved gain matrix
	disp(['K_', num2str(it), '=']);
	disp(K);
	k_save=[k_save,norm(K-K0)];     % Keep track of the control gains
end

% Post-learning simulation
[tt,xx]=ode45(@mysys,[t(end) 50],X(end,:)');

% Keep track of the post-learning trajectories
t_final=[t_save;tt];
x_final=[x_save;xx];

% The following nested function gives the dynamics of the sytem. Also,
% integraters are included for the purpose of data collection.
	function dX = mysys(t,X)
		%global A B xn un i1 i2 K flag
		x=X(1:xn);
		
		if ~ifLearned;   % See if learning is stopped
			u = sum(sin(expl_noise_freq*t),2)/100;
		else
			u = -K*x; % Exploitation
		end
		
		dx  = A*x+B*u;
		dxx = kron(x',x')';
		dux = kron(x',u')';
		dX  = [dx;dxx;dux];
	end

%% Plotting results
figure(1)
subplot(211)
plot(0:length(p_save)-1,p_save,'o',[0:length(p_save)-1],p_save);
ylim([-max(p_save)*0.2 max(p_save)*1.2]);
legend('||P_k-P^*||')
xlabel('Number of iterations')

subplot(212)
plot(0:length(k_save)-1,k_save,'^',[0:length(k_save)-1],k_save)
ylim([-max(k_save)*0.2 max(k_save)*1.2]);
legend('||K_k-K^*||')
xlabel('Number of iterations')

figure(2)
plot(t_final,x_final(:,1:6),'Linewidth',2)
legend('x_1','x_2','x_3','x_4','x_5','x_6')
xlabel('Time (sec)')

figure(3)
plot(t_final,3.6*x_final(:,1),'k-.', ...
	t_final, x_final(:,6),'-','Linewidth',2)
legend('y_1 (MAF)','y_2 (MAP)')
xlabel('Time (sec)')


end
