functions { 
  vector fbieri(vector y, vector theta,
                              real[] x_r, int[] x_i){
 real x = y[1];
 real a = theta[1];
 real b = theta[2];
 real tmin = theta[3];
 real tmax = theta[4];
 real val;
 vector[1] y_x;
 val = a*(x)-a*tmin-pow(b,x-tmax);
 y_x[1] = val;
return y_x; }

  real my_normal_lpdf(real y, real mu, real sigma) {
  real seval;
    seval= -0.5*log(2*pi())-log(sigma)-(0.5*(y-mu)*(y-mu))/(sigma^2);
    return seval;
  }

}
data {
int<lower=0> N;
int<lower=0> NP; 
vector[NP] xpred;
vector[N] y;
vector[N] x;
real status[N];
}
transformed data{
vector[N] ey=exp(y);
real l1=min(x)-min(x)/20;
real l2=max(x)-max(x)/20;
vector[1] y_guess1;
vector[1] y_guess2;
real x_r[0];
int x_i[0];
y_guess1[1]= l1;
y_guess2[1]= l2;
}
parameters {
  real <lower=0, upper=1>  a;
real <lower=1>  b;
real <lower=0> tmin;
real <lower=tmin> tmax;
real <lower=0>  sigmasq;
}
transformed parameters {
real  sigma;
 vector[4] theta;
 theta[1]=a;
 theta[2]=b;
 theta[3]=tmin;
 theta[4]=tmax;
 sigma = sqrt(sigmasq);

}
model {
real ypred[N];
  real count=0;
for (i in 1:N)
{ypred[i] = a*(x[i])-a*tmin-pow(b,x[i]-tmax);
y[i] ~ normal(ypred[i], sigma);
count=x[i];
}
 tmax ~ gamma(.1, .01); 
 tmin ~ gamma(.1, .01); 
a ~ beta(1,1);
b ~ gamma(.2, .1);
sigmasq ~ inv_gamma(1e-3, 1e-3);

}
generated quantities { 
real log_lik[N];
real sscale2[NP];
real dev;
real tdmax;
real yeval[N];
real shape[N];
real ypred[NP];
real mu[NP];
vector[1] gmin;
vector[1] xmin;
vector[1] xmax;
vector[1] gmax;
gmin[1]=tmin;
gmax[1]=2*tmax;
for (i in 1:N) {
  shape[i] = a*(x[i])-a*tmin-pow(b,x[i]-tmax);  

log_lik[i] = normal_lpdf(y[i] |shape[i] , sigma);
yeval[i] = normal_rng(shape[i] , sigma);} 


for (j in 1:NP) {
sscale2[j] = a*(xpred[j])-a*tmin-pow(b,xpred[j]-tmax);  
if(is_nan(sscale2[j]))
{
  ypred[j]=0;
  sscale2[j]=0;
}else{
  ypred[j] = normal_rng(sscale2[j],sigma);
}
mu[j]=sscale2[j];}
dev = -2*sum(log_lik[]);
tdmax = (log(a)-log(log(b)))/log(b)+tmax;
xmin = algebra_solver(fbieri,gmin , theta, x_r, x_i, 1e-70, 1e+1, 1e+70); 
xmax = algebra_solver(fbieri, gmax , theta, x_r, x_i, 1e-70, 1e+1, 1e+70);
} 
