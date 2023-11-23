clc
clear;

% Code implemented from the paper: Deforestation Analysis Using Remote
% Sensing

image = imread('PIA11420~orig.jpg');
rgb_image = image;

%% Color transformation to L*a*b color space and HSV
rgbToLab = makecform('srgb2cmyk');
Lab = applycform(rgb_image, rgbToLab);

ImHSV = rgb2hsv(image);

%% Color transformation to HSI
X = image;
XX = im2double(X);

r = XX(:,:,1);
g = XX(:,:,2);
b = XX(:,:,3);

th = acos((0.5*((r-g)+(r-b)))./((sqrt((r-g).^2+(r-b).*(g-b)))+eps));

H = th;
H(b>g) = 2*pi-H(b>g);
H = H/(2*pi);

S = 1-3.*(min(min(r,g), b))./(r + g + b + eps);
I = (r+g+b) / 3;

ImHSI = cat(3, H,S,I); %HSI Color Space Transformation

%% Extracting the H, S, I channels
H = ImHSI(:,:,1);
S = ImHSI(:,:,2);
I = ImHSI(:,:,3);

%% Extracting Channels of L*a*b 
L = Lab(:,:,1);
A = Lab(:,:,2);
B = Lab(:,:,3);

%% Extracting H, S, V channels
h = ImHSV(:,:,1);
s = ImHSV(:,:,2);
v = ImHSV(:,:,3);

%% FCM on the image
IM = mat2gray(L);
data = reshape(IM, [], 1);
[center, member] = fcm(data, 3, 3);
[x, level] = max(member, [], 1);
IMout = reshape(center(level), size(L));

IM_fcm = im2bw(IMout);

[m, n] = size(IM_fcm);

total = bwarea(IM_fcm);
Area = total/(m*n);

%disp(Area);

figure;
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(IM_fcm); title('FCM segmented');

