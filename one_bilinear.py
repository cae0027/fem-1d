
 1
 2
 3
 4
 5
 6
 7
 8
 9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
import numpy as np

def oned_bilinear(kernel,phi,test,w_g):
#-----------------------------------------------------------------------
#  oned_bilinear.m - routine to compute \int{ kernel*phi*test }
#
#  Copyright (c) 2001, Jeff Borggaard, Virginia Tech
#  Version: 1.0
#
# Usage:    M = oned_bilinear(kernel, phi, test, w_g)
#
#  Variables:     kernel
#                        Kernel function in the integral evaluated
#                        at the Gauss points
#
#                 phi
#                        matrix of element test functions evaluated
#                        at the Gauss points (dim: n_gauss, n_dof)
#
#                 test
#                        matrix of test functions evaluated at the
#                       Gauss points (dim: n_gauss, n_test)        
#
#                 w_g
#                        Column vector of Gauss weights
#-----------------------------------------------------------------------
#  test this
#  n_test = size(test);
#  n_gauss= size(w_g)
#  n_dof = size(phi);

# M = np.zeros(n_test,n_dof);
# for i=1:n_test
#    for j=1:n_dof
#       M[i,j] = ( kernel' .* test[:,i]' )*( phi[:,j] .* w_g );
#    end
# end

#  n_test = size(test);
#  n_gauss= size(w_g)
#  n_dof = size(phi);

# M = np.zeros(n_test,n_dof);
# for i=1:n_test
#    for j=1:n_dof
#       M[i,j] = test[:,i]'*( phi[:,j] .* kernel .* w_g );
#    end
# end

#  finally, try this
# M = test' * ( phi .* kernel .* w_g );

#  Vectorized version is more efficient (even for small vector lengths)
 M = (np.dot(np.transpose(test),np.diag(kernel*w_g)))*phi

