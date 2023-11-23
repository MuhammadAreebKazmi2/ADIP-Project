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

%% Displaying the L*, a*, b* channels
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(L); title('L* channel');
subplot(223), imshow(A); title('a* channel');
subplot(224), imshow(B); title('b* channel');

%% Displaying HSV Channel
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(h); title('H channel');
subplot(223), imshow(s); title('S channel');
subplot(224), imshow(v); title('V channel');

%% Displaying HSI Channel
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(H); title('H channel');
subplot(223), imshow(S); title('S channel');
subplot(224), imshow(I); title('I channel');

%% Applyy Otsu Method segmentation in the L*, a*, b* channels
level = graythresh(L);
binary = im2bw(L, level);
Segmented_L = imfill(binary, "holes"); % L* channel segmentation

level = graythresh(A);
binary = im2bw(A, level);
Segmented_A = imfill(binary, "holes"); % a* channel segmentation

level = graythresh(B);
binary = im2bw(B, level);
Segmented_B = imfill(binary, "holes"); % b* channel segmentation

% Displaying transformed color space images
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(Segmented_L); title('L* segmented');
subplot(223), imshow(Segmented_A); title('a* segmented');
subplot(224), imshow(Segmented_B); title('b* segmented');

%% Applying Otsu Method segmentation in HSV Channel
level = graythresh(h);
binary = im2bw(h, level);
Segmented_h = imfill(binary, "holes"); % H channel segmentation

level = graythresh(s);
binary = im2bw(s, level);
Segmented_s = imfill(binary, "holes"); % S channel segmentation

level = graythresh(v);
binary = im2bw(v, level);
Segmented_v = imfill(binary, "holes"); % V channel segmentation

% Displaying transformed color space images
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(Segmented_h); title('H segmented');
subplot(223), imshow(Segmented_s); title('S segmented');
subplot(224), imshow(Segmented_v); title('V segmented');

%% Applying Otsu Method segmentation in HSI Channel
level = graythresh(H);
binary = im2bw(H, level);
Segmented_H = imfill(binary, "holes"); % H channel segmentation

level = graythresh(S);
binary = im2bw(S, level);
Segmented_S = imfill(binary, "holes"); % S channel segmentation

level = graythresh(I);
binary = im2bw(I, level);
Segmented_I = imfill(binary, "holes"); % I channel segmentation

% Displaying transformed color space images
figure,
subplot(221), imshow(rgb_image); title('Original Image');
subplot(222), imshow(Segmented_H); title('H segmented');
subplot(223), imshow(Segmented_S); title('S segmented');
subplot(224), imshow(Segmented_I); title('I segmented');
