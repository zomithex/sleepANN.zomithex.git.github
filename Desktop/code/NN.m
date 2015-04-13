function [Y,Xf,Af] = myNeuralNetworkFunction(X,~,~)
%ANN function for sleep scoring
%
% 
% 
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
% 
%   X = 1xTS cell, 1 inputs over TS sleep timsteps
%   Each X{1,ts} = 2xQ matrix, input #1 at sleep timestep ts.
% 
% and returns:
%   Y = 1xTS cell of 1 outputs over TS sleep timesteps.
%   Each Y{1,ts} = 2xQ matrix, output #1 at sleep timestep ts.
% 
% where Q is number of samples (or series) and TS is the number of sleep timesteps.

%#ok<*RPMT0>

  % ===== NEURAL NETWORK CONSTANTS =====
  
  % Input 1
  x1_step1_xoffset = [0;0];
  x1_step1_gain = [2;2];
  x1_step1_ymin = -1;
  
  % Layer 1
  b1 = [4.4270912904368238;-3.4144953278571499;-2.457934698501139;-1.4753119377534236;-0.46643196360787997;0.30638296666560721;-1.1838836761256102;2.0041885667111141;-3.4631700916273762;4.3844733712506025];
  IW1_1 = [-3.1176785049364679 -3.1432576627685234;3.7595415687242499 -2.4125929742637355;3.0614060886250254 -3.2012517386137507;1.7958087346028919 -4.0472182837934154;3.012983788440589 3.2083614709049546;3.84677540121942 2.3764781337126726;-1.9512439329946258 -4.1105502415237227;0.98965413643904798 4.6401579950474581;-2.5743131478881298 -3.5959280540354568;4.2863440393688919 -1.3043078188896511];
  
  % Layer 2
  b2 = [-0.19753778647917555;-0.28035787926846129];
  LW2_1 = [0.055323143472455781 1.792415669082448 1.1875046311770816 1.5011087016394804 0.039312347428517212 1.2648353686018221 0.019159267849547947 -0.966459512302412 -0.59950694429010054 0.56003775900164432;0.23882734701510491 -1.8282706576977017 -1.3168034381769376 -0.78975301969892198 0.02211117053153018 -0.59523400694495054 -1.4559404343426072 1.8826755332972229 0.45081389957842288 -0.60987976140605249];
  
  % ===== SIMULATION ========
  
  % Format Input Arguments
  isCellX = iscell(X);
  if ~isCellX, X = {X}; end;
  
  % Dimensions
  TS = size(X,2); % sleep timesteps
  if ~isempty(X)
    Q = size(X{1},2); % samples/series
  else
    Q = 0;
  end
  
  % Allocate Outputs
  Y = cell(1,TS);
  
  % Time loop
  for ts=1:TS
  
    % Input 1
    Xp1 = mapminmax_apply(X{1,ts},x1_step1_gain,x1_step1_xoffset,x1_step1_ymin);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = softmax_apply(repmat(b2,1,Q) + LW2_1*a1);
    
    % Output 1
    Y{1,ts} = a2;
  end
  
  % Final Delay States
  Xf = cell(1,0);
  Af = cell(2,0);
  
  % Format Output Arguments
  if ~isCellX, Y = cell2mat(Y); end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings_gain,settings_xoffset,settings_ymin)
  y = bsxfun(@minus,x,settings_xoffset);
  y = bsxfun(@times,y,settings_gain);
  y = bsxfun(@plus,y,settings_ymin);
end

% Competitive Soft Transfer Function
function a = softmax_apply(n)
  nmax = max(n,[],1);
  n = bsxfun(@minus,n,nmax);
  numer = exp(n);
  denom = sum(numer,1); 
  denom(denom == 0) = 1;
  a = bsxfun(@rdivide,numer,denom);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end
