data {
int<lower=0> N;
real y[N];
real x[N];
real status[N];
int<lower=0> NP; 
real xpred[NP];
}
parameters {
  real <lower=0> tmax;
real  <lower=0, upper=tmax> tmin;
real <lower=0> a;
real  <lower=0> sigmasq;
}
transformed parameters {
real <lower=0>  sigma;
sigma = sqrt(sigmasq);
}
model {
real ypred[N];


for (i in 1:N)
{
  
  if (x[i] > tmin && x[i] < tmax ){ 
  ypred[i] = exp(-a + log(x[i])+log(x[i]-tmin)+0.5*log(tmax-x[i]));  
  target+=normal_lpdf(y[i]|ypred[i], sigma);
  
} else ypred[i]=0;
}
    target+=gamma_lpdf(tmax|0.01, 0.001);
    target+=gamma_lpdf(tmin|0.01, 0.01);
    target+=gamma_lpdf(a|0.1, 0.01);
    target+=inv_gamma_lpdf(sigmasq|0.001, 0.001);
}
generated quantities { 
real log_lik[N];
real yeval[N];
real sscale2[NP];
real scale[N];
real ypred[NP];
real mu[NP];
real dev;
real alpha;
real  <lower=0> tdmax;
alpha = exp(-a);
for (i in 1:N) {
if ( x[i] > tmin && x[i] < tmax )
{
scale[i] = alpha*(x[i])*(x[i]-tmin)*sqrt(tmax-x[i]);    
log_lik[i] = normal_lpdf(y[i] |scale[i], sigma);
yeval[i] = normal_rng(alpha*(x[i])*(x[i]-tmin)*sqrt(tmax-x[i]),sigma);
}
else
{
log_lik[i] = 0;
yeval[i] = 0;
scale[i] = 0;
}
}
for (j in 1:NP) {
sscale2[j] = alpha*(xpred[j])*(xpred[j]-tmin)*sqrt(tmax-xpred[j]);  
if(is_nan(sscale2[j]))
{
  ypred[j]=0;
  sscale2[j] =0;
}else{
  ypred[j] = normal_rng(sscale2[j],sigma);
}
mu[j]=sscale2[j];
}
dev = -2*sum(log_lik[]);
tdmax = ((4*tmax+3*tmin)+sqrt(pow((4*tmax+3*tmin),2)-40*tmin*tmax))/10;
}
