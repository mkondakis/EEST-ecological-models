data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
   int<lower=0> NP; 
vector[NP] xpred;
  }
    transformed data {
    vector[N] ey=exp(y);
  }
  parameters {
  real <lower=0> tmax;
  real <lower=0> a;
  real  <lower=0> mm;
  real  <lower=mm, upper=4> nn;
  real <lower=4, upper=tmax> tmin;
  real <lower=2> shape;
  }
  model {
  real ypred[N];
  real ag;
  real bg;
  real cntr=0;
  real count=0;
  for (i in 1:N)
  {
    ypred[i] = exp(-a+nn*log(x[i]-tmin)+mm*log(tmax-x[i]));
    if(!is_nan(ypred[i]) &&  ypred[i]>0) target+= inv_gamma_lpdf(y[i]| shape , (shape-1)*(ypred[i])); else ypred[i]=0;
  }
        tmax ~ gamma(.01,.01);
    nn ~ gamma(0.1, 0.1);
      a ~ gamma(0.1, 0.01);
   tmin ~ gamma(.01,.01);
       mm ~ gamma(0.1, 0.1);
    shape ~ gamma(.01, .001); 
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
   real mu[NP];
real ypred[NP];
  count=0;
  for (i in 1:N) 
  {
    scale[i] = exp(-a+nn*log(x[i]-tmin)+mm*log(tmax-x[i]));

    if (!is_nan(scale[i]) &&  scale[i]>0 )//if (x[i] > tmin && x[i] < tmax ) 
    {
    log_lik[i] = inv_gamma_lpdf(y[i] |shape , (shape-1)*scale[i]);
    yeval[i] = log(inv_gamma_rng(shape , (shape-1)*scale[i]));
    } else
    {
    scale[i] = 0;  
    log_lik[i] = 0;
    count +=1;
    yeval[i] = 0;
    }  
  }
   for (j in 1:NP) {
sscale2[j] = exp(-a+ nn*log(xpred[j]-tmin)+mm*log(tmax-xpred[j]));
  if ( !is_nan(sscale2[j]) &&  sscale2[j]>0 )  
{
    ypred[j] = (inv_gamma_rng(shape,(shape-1)*sscale2[j]));
     mu[j] = sscale2[j];
}else{
  ypred[j]=0;
    sscale2[j]=0;
    mu[j]=0;
  
}}
  alpha= exp(-a);
  dev = -2*sum(log_lik[]);
  tdmax = (nn*tmax+mm*tmin)/(nn+mm);
}
