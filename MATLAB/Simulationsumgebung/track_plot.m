%% prepare the data
% load the track data and sort for x- and y-values
load('track.mat');
load('track_pos.mat');
trck_data=trck_pos.Data;
yt=trck_data(:,1);
xt=-trck_data(:,2);
x = data(1:3:end);
y = data(2:3:end);

% outer track limit
x_out = x(1:2:end);
y_out = y(1:2:end);
x_out(1:3:end) = [];
y_out(1:3:end) = [];

% inner track limit
x_in = x(2:2:end);
y_in = y(2:2:end);
x_in(1:3:end) = [];
y_in(1:3:end) = [];

% calculate the center line
median_x = (x_in + x_out)/2;
median_y = (y_in + y_out)/2;

%% plot
figure
axis([-150 250 -400 500]);
grid on
hold on

plot(x_in,y_in,'k-');
plot(x_out,y_out,'k-');
plot(median_x, median_y,'r--');
plot(0,0,'bx');
plot(xt,yt)

%% error
% rearrange vectors to start at the same position
[v1,i1]=min(xt);
[v2,i2]=min(median_x);
xt_r=zeros(size(xt));
yt_r=zeros(size(yt));
median_xr=zeros(size(median_x));
median_yr=zeros(size(median_y));
xt_r(1:length(xt)-i1+1)=xt(i1:end);
xt_r(length(xt)-i1+1:end)=xt(1:i1);
yt_r(1:length(yt)-i1+1)=yt(i1:end);
yt_r(length(yt)-i1+1:end)=yt(1:i1);
median_xr(1:length(median_x)-i2+1)=median_x(i2:end);
median_xr(length(median_x)-i2+1:end)=median_x(1:i2);
median_yr(1:length(median_y)-i2+1)=median_y(i2:end);
median_yr(length(median_y)-i2+1:end)=median_y(1:i2);

% reduce samples
xt_r=xt_r(1:18:end);
yt_r=yt_r(1:18:end);

% calculate error
nm=min(length(xt_r),length(median_xr));
median_xr=median_xr(:)';
median_xr=median_xr(1:nm);
median_yr=median_yr(:)';
median_yr=median_yr(1:nm);
xt_r=xt_r(:);
xt_r=xt_r(1:nm)';
yt_r=yt_r(:);
yt_r=yt_r(1:nm)';

error=[abs(median_xr-xt_r);abs(median_yr-yt_r)];
t=trck_pos.Time(1:18:end);
t=t(1:nm);

figure()
subplot(2,1,1)
plot(t,median_xr,'r--',t,xt_r,'b');
ylabel('compare x to median x')
xlabel('time [s]');
title('compare x and y')
legend('median','track sim')

subplot(2,1,2)
plot(t,median_yr,'r--',t,yt_r,'b');
ylabel('compare y to median y')
xlabel('time [s]');

figure()
subplot(2,1,1)
plot(t,error(1,:));
ylabel('error in x')
xlabel('time [s]');
title('error in x and y direction')

subplot(2,1,2)
plot(t,error(2,:));
ylabel('error in y')
xlabel('time [s]');
