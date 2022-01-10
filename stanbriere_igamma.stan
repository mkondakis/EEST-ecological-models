data {
int<lower=0> N;
real y[N];
real x[N];
real status[N];
int<lower=0> NP; 
vector[NP] xpred;
}
parameters {
  
real <lower=0> tmax;
real  <lower=0, upper=tmax> tmin;
real <lower=0> a;
  real <lower=2> shape;
}
model {
real mu[N];

for (i in 1:N)
{
    
  mu[i] = x[i]*(x[i]-tmin)*sqrt(tmax-x[i])*exp(-a);  
  if(!is_nan(mu[i]) &&  mu[i]>0) target +=  inv_gamma_lpdf(y[i]|shape , (shape-1)*mu[i]); else mu[i]=0;

}
  tmax ~ gamma(.01, .01);
  tmin ~ gamma(.01,.01);
  a ~ gamma(0.1, 0.01);
  shape ~ gamma(.01,.01);
}
 generated quantities { 
  real log_lik[N];
  real yeval[N];
  real dev;
  real tdmax;
  real scale[N];
  real alpha;
  int  count;
  real sscale2[NP];
  real ypred[NP];
  real mu[NP];

  count=0;
  alpha= exp(-a);

  for (i in 1:N) 
  {

              scale[i] = exp(-a)*x[i]*(x[i]-tmin)*sqrt(tmax-x[i]);
       if (!is_nan(scale[i]) &&  scale[i]>0 )//if (x[i] > tmin && x[i] < tmax )
    {

    scale[i]=(shape-1)*scale[i];
    log_lik[i] = inv_gamma_lpdf(y[i]|shape,scale[i]);
    yeval[i] = inv_gamma_rng(shape ,scale[i]);
    } else
    {
    scale[i] = 0;  
    log_lik[i] = 0;
    count +=1;
    yeval[i] = 0;
    }  
  }
  
  for (j in 1:NP) {
        sscale2[j] = exp(-a)*xpred[j]*(xpred[j]-tmin)*sqrt(tmax-xpred[j]);
  if ( !is_nan(sscale2[j]) &&  sscale2[j]>0 )  // if (xpred[j] > tmin && xpred[j] < tmax ) 
{
  
    ypred[j] = (inv_gamma_rng(shape,(shape-1)*sscale2[j]));
     mu[j] = sscale2[j];
}else{
  ypred[j]=0;
  sscale2[j] =0;
    mu[j] =0;
}}
  
  dev = -2*sum(log_lik[]);
tdmax = ((4*tmax+3*tmin)+sqrt(pow((4*tmax+3*tmin),2)-40*tmin*tmax))/10;
} 
