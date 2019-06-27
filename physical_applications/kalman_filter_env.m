clear all
clc
%close all

%% generate data
L           = 1000;
T           = 1:L;
y1          = [0,50,-9.8]';
y           = [y1];
dT          = .01;
M           = [1,dT,.5*dT^2; 0, 1, dT; 0, 0, 1];
for i = 1:L
    next        = M*y(:,i);
    y           = [y,next];
end
yobs        = y + 20*randn(size(y));

plot(yobs(1,:))

%% end mark section
y_true = y;

y      = yobs;
sys.M  = [1,dT,.5*dT^2; 0, 1, dT; 0, 0, 1];         %state update
sys.Q  = [1 0 0; 0  0.2 0; 0 0 .1];                 %prediction noise
sys.x0 = y(:,1);                                    %start position
sys.P0 = [1 0 0; 0 1 0; 0 0 1];

sys.H = eye(3);
sys.R = [1 0 0; 0 1 0 ; 0 0 2]*10; %observation noise
%%
[x,P,K] = KalmanPG(y,sys);
%sys.A = eye(2);sys.B = eye(2);
%sys.Ex = [0 0]; sys.Px = [1 0; 0 0.01];
%[x,P,K] = kalmanCAG(y,sys);

% T=iday;
T       = [T,T(end)+dT];
figure(1)
subplot(311)
plot(T,y(1,:),'b','MarkerSize',10)
hold on
grid on
plot(T,x(1,:),'r','MarkerSize',15)
plot(T,y_true(1,:),'k')
legend('Observed','Kalman Est','True')
ylabel('Position (m)')


subplot(312)
plot(T,y(2,:))
hold on
grid on
plot(T,x(2,:),'r')
plot(T,y_true(2,:),'k')
ylabel('Velocity (m/s)')

subplot(313)
plot(T,y(3,:))
hold on
grid on
plot(T,x(3,:),'r')
plot(T,y_true(3,:),'k')
xlabel('time index')
ylabel('Acceleration (m/s^2)')
