function [ I ] = AnisotropicDiffusion(I, iterations)

% initialize the values (i+1,j), (i-1,j), (i,j+1), (i,j-1)
Ixp = zeros( size( I ) );
Ixm = zeros( size( I ) );
Iyp = zeros( size( I ) );
Iym = zeros( size( I ) );
Ixyp = zeros( size( I ) );
Ixym = zeros( size( I ) );

dt = 0.25;   % timestep

useNeumann = true; % use Neumann boundary conditions, otherwise Dirichlet
dSubfigure = 1;

for iI=0:iterations

  % compute the neighboring values 
  Ixp(1:end-1,:) = I(2:end,:); Ixm(2:end,:) = I(1:end-1,:);
  Iyp(:,1:end-1) = I(:,2:end); Iym(:,2:end) = I(:,1:end-1);

  % use Neumann boundary conditions (zero derivative at boundary)
  Ixp(end,:) = I(end,:); Ixm(1,:) = I(1,:);
  Iyp(:,end) = I(:,end); Iym(:,1) = I(:,1);

  % compute the derivatives
  Ixx = Ixp-2*I+Ixm;
  Iyy = Iyp-2*I+Iym;
  Ix = (Ixp-Ixm)/2;
  Iy = (Iyp-Iym)/2;
  Ixyp(:,1:end-1) = Ix(:,2:end);
  Ixym(:,2:end) = Ix(:,1:end-1);
  Ixy = (Ixyp-Ixym)/2;
  
  % geometric heat equation
  dEps = 0.001;
  nDenominator = Ix.^2 + Iy.^2 + dEps^2;
  nNumerator = (Iy.^2).*Ixx - 2*(Ixy.*Ix.*Iy) + (Ix.^2).*Iyy;
  
  % compute the new value by averaging
  Ikp1 = I + dt*(nNumerator./nDenominator); % anisotropic diffusion
  
  if ( useNeumann )
    I = Ikp1;
  else  % using Dirichlet boundary conditions (ie., fix the boundary)
    I(2:end-1,2:end-1) = Ikp1(2:end-1,2:end-1);
  end
end

end

