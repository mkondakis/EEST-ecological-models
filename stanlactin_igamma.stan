functions { 
   vector flactin(vector y, vector theta,
 real[] x_r, int[] x_i){
 real x = y[1];
 real l = theta[1];
 real ro = theta[2];
 real a = theta[3];
 real del = theta[4];
 real val;
 vector[1] y_x;
 val = -l +  exp(ro*x) - a*exp(del*x);
 y_x[1] = val;
return y_x;                              
}
   real Sinv_lpdf(vector y, vector x, real shape, vector sc, real a, real l, real ro , real del) {
    real value;
    int N;
    N=num_elements(y);
    value = shape * N * log(shape-1) - shape*N*l + shape*sum(exp(x*ro)) - shape*a*sum(exp(x*del)) - N*lgamma(shape) - (shape+1)*sum(log(y)) - sum(sc ./ y);
    return value;
  }
    real Sinvs_lpdf(real y, real x, real shape, real sc, real a, real l, real ro , real del) {
    real value;
    value = shape *  log(shape-1) - shape*l + shape*exp(x*ro) - shape*a*exp(x*del) - lgamma(shape) - (shape+1)*(log(y)) - (sc / y);
    return value;
  }
    real Sin_lpdf(real y, real x, real shape, real sc, real a, real l, real ro , real del) {
    real value;
    value = shape *  log(shape-1) + shape*log(-l + exp(x*ro) - a*exp(x*del)) - lgamma(shape) - (shape+1)*(log(y)) - ((shape-1)*sc / y);
    return value;
  }  }
data {
  int<lower=0> N;
vector[N] y;
vector[N] x;
int<lower=0> NP;
vector[NP] xpred;

}
transformed data{
vector[N] ey=exp(y);
real l1=-min(x)-min(x)/2;
real l2=max(x)+max(x)/2;
vector[1] y_guess1;
vector[1] y_guess2;
real x_r[0];
int x_i[0];
y_guess1[1]= l1;
y_guess2[1]= l2;
}
parameters {

  real <lower=0, upper=1> del; 
  real <lower=0, upper=del>  ro;
   real  <lower=((log(ro)-log(del))/(ro-del))> tmax;
  real  <lower=-1, upper=1> l; 
  real <lower=2> shape;
}
transformed parameters{
 vector[4] theta;
 real <lower=0, upper=(ro/del)> a;
 a=exp((ro-del)*tmax);
 theta[1]=l;
 theta[2]=ro;
 theta[3]=a;
 theta[4]=del;
}
model {
vector[N] mu;
real count;

for (i in 1:N)
{
  mu[i] =  (-l+exp(ro*x[i])-a*exp(del*x[i]));

  if(is_nan(mu[i]) ) target += inv_gamma_lpdf( y[i] |shape, (shape-1)*mu[i]); else mu[i]=0;
 
 
}

target += gamma_lpdf(tmax|0.1, 0.01);
del ~ beta(1, 1);
ro ~ beta(1, 1);
shape ~ gamma(.1, .01); 
}
generated quantities { 
real log_lik[N];
real yeval[N];
real mu[NP];
real ypred[NP];
real sscale2[NP];
real sscale[N];
real dev;
real  <lower=0, upper=tmax> tdmax;
vector[1] xmin;
vector[1] xmax;
real s2cale=0;
xmin = algebra_solver(flactin, y_guess1, theta, x_r, x_i, 1e-10, 1e+1, 1e+50);
xmax = algebra_solver(flactin, y_guess2, theta, x_r, x_i, 1e-10, 1e+1, 1e+50);
tdmax = tmax-log(ro/del)/(ro-del);

for (i in 1:N) {
sscale[i] = (-l+exp(ro*x[i])-a*exp(del*x[i]));

if (!is_inf(sscale[i]) && sscale[i]>0)

{        
  log_lik[i] = inv_gamma_lpdf( y[i] |shape, (shape-1)*sscale[i]);
  if (sscale[i]>0) yeval[i] = inv_gamma_rng(shape,(shape-1)*sscale[i]); else yeval[i]=0;


}else 
{
     yeval[i]=0;
     log_lik[i]=0;
     sscale[i]=0;
}
}
for (j in 1:NP) {
  sscale2[j] = (-l+exp(ro*xpred[j])-a*exp(del*xpred[j]));
 
if ( sscale2[j]>0 && (!is_inf(sscale2[j])) )
  {
    mu[j]=sscale2[j];
    ypred[j]=inv_gamma_rng(shape,(shape-1)*sscale2[j]);
  } else 
  {
    sscale2[j]=0;
    ypred[j]=0;
    mu[j]=0;
  }
}
dev = -2*sum(log_lik[]);
} 
