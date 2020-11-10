function res = circles()

tic
%
% Size of my grid -- higher values => higher accuracy.
%
ngrid = 5000;

xc = [1.6417233788 -1.4944608174  0.6110294452  0.3844862411 -0.2495892950  1.7813504266 -0.1985249206 -1.7011985145 -0.4319462812  0.2178372997 -0.6294854565  1.7952608455  1.4168575317  1.4637371396 -0.5263668798 -1.2197352481 -0.1389358881  1.5293954595 -0.5258728625 -0.1403562064  0.8055826339 -0.6311979224  1.4685857879 -0.6855727502  0.0152957411];
yc = [1.6121789534  1.2077959613 -0.6907087527  0.2923344616 -0.3832854473  1.6178237031 -0.8343333301 -0.1263820964  1.4104420482 -0.9499557344 -1.3078893852  0.6281269104  1.0683357171  0.9463877418  1.7315156631  0.9144146579  0.1092805780  0.0030278255  1.3782633069  0.2437382535 -0.0482092025  0.7184578971 -0.8347049536  1.6465021616  0.0638919221];
r  = [0.0848270516  1.1039549836  0.9089162485  0.2375743054  1.0845181219  0.8162655711  0.0538864941  0.4776976918  0.7886291537  0.0357871187  0.7653357688  0.2727652452  1.1016025378  1.1846214562  1.4428514068  1.0727263474  0.7350208828  1.2472867347  1.3495508831  1.3804956588  0.3327165165  0.2491045282  1.3670667538  1.0593087096  0.9771215985];
r2 = r .* r;

ncircles = length(xc);

%
% Compute the bounding box of the circles.
%
xmin = min(xc-r);
xmax = max(xc+r);
ymin = min(yc-r);
ymax = max(yc+r);

%
% Keep a counter.
%
inside = 0;

%
% For every point in my grid.
%
for x = linspace(xmin,xmax,ngrid)
    for y = linspace(ymin,ymax,ngrid)
        if any(r2 > (x - xc).^2 + (y - yc).^2)
            inside = inside + 1;
        end
    end
end

box_area = (xmax-xmin) * (ymax-ymin);

res = box_area * inside / ngrid^2;
toc

end
