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
    //real rat;
    int N;
    N=num_elements(y);
    value = shape * N * log(shape-1) - shape*N*l + shape*sum(exp(x*ro)) - shape*a*sum(exp(x*del)) - N*lgamma(shape) - (shape+1)*sum(log(y)) - sum(sc ./ y);
    return value;
  }
    real Sinvs_lpdf(real y, real sigma, real sc) {
    real value;
    real pi2=sqrt(2*pi());
    value = -log(pi2*sigma)-(1.0/2.0)*(1/sigma)*(y-sc)*(y-sc)*(1/sigma);
    return value;
  }
  }
data {
int<lower=0> N;
int<lower=0> NP;  
vector[NP] xpred;
vector[N] y;
vector[N] x;
vector[N] status;
}
transformed data{
vector[N] ey=exp(y);
real l1=min(x)-min(x)/2;
real l2=max(x)+max(x)/2;
vector[1] y_guess1;
vector[1] y_guess2;
real x_r[0];
int x_i[0];
int M=10;
//real st=1e-3;
y_guess1[1]= l1;
y_guess2[1]= l2;
}
parameters {
    real <lower=0, upper=1> del; 
  real <lower=0, upper=del>  ro; 
       real  <lower=((log(ro)-log(del))/(ro-del)),upper=35> tmax;
  real l; 
real <lower=0>  sigmasq;
}
transformed parameters{
  real sigma;
   real <lower=0, upper=(ro/del)> a;
 vector[4] theta;
  a=exp((ro-del)*tmax);
 theta[1]=l;
 theta[2]=ro;
 theta[3]=a;
 theta[4]=del;
  sigma = sqrt(sigmasq);

}
model {
vector[N] mu;
real count;
real cntr; 
real dst;
real ag;
real bg;
for (i in 1:N)
  {
    mu[i] = -l +  exp(ro*x[i]) - a*exp(del*x[i]); //exp((ro-del)*tmax) is equivalent to a
  y[i] ~ normal(mu[i], sigma);
  }
target += gamma_lpdf(tmax|0.1, 0.01);
del ~ beta(1, 1);
l ~ normal(0,100);
ro ~ beta(1, 1);
sigmasq ~ inv_gamma(1e-3, 1e-3);

}
generated quantities { 
real log_lik[N];
real yeval[N];
real sscale[N];
real ypred[NP];
real sscale2[NP];
real mu[NP];
real dev;
real  <lower=0, upper=tmax> tdmax;
vector[1] xmin;
vector[1] xmax;
real s2cale=0;
real delta;
delta=1/del;
xmin = algebra_solver(flactin, y_guess1, theta, x_r, x_i, 1e-10, 1e+1, 1e+50);
xmax = algebra_solver(flactin, y_guess2, theta, x_r, x_i, 1e-10, 1e+1, 1e+50);

tdmax = tmax-log(ro/del)/(ro-del);

for (i in 1:N) {
  log_lik[i]=0;
     yeval[i]=0;

sscale[i] = -l +  exp(ro*x[i]) - a*exp(del*x[i]);
   
 if(!is_nan(sscale[i]) && !is_inf(sscale[i])){
   log_lik[i]=normal_lpdf( y[i] |sscale[i],sigma);
    yeval[i] = normal_rng(sscale[i],sigma);

}else 
{sscale[i]=0;
log_lik[i]=0;
yeval[i]=0;
}
  
}
for (j in 1:NP) {
sscale2[j] = -l +  exp(ro*xpred[j]) - a*exp(del*xpred[j]);  
if(is_nan(sscale2[j]))
{
  sscale2[j]=0;
  ypred[j]=0;
}else{
  ypred[j] = normal_rng(sscale2[j],sigma);
}
mu[j]= sscale2[j];
}
dev = -2*sum(log_lik[]);

}
