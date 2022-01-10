data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  real status[N];
  int<lower=0> NP; 
vector[NP] xpred;
  }
  parameters {
      real  <lower=4> tmin;
 real  <lower=0> nn;
   real  <lower=0> mm;
  real <lower=35> tmax;
  real <lower=0> a;
    //real <lower=32.5, upper=35> tmax;
    ////////////////////////
  //      real <lower=0> a;
  //  real  <lower=0, upper=1> nn;
  //  real  <lower=0, upper=1> mm;
  //          real <lower=0> tmin;
  //        real <lower=tmin, upper=max(x)> tmax;
  real <lower=0> sigmasq;
  }
   transformed parameters {
  real  sigma;
  sigma = sqrt(sigmasq);
  }
  model {
  real ypred[N];
  for (i in 1:N)
  {
 if (x[i] > tmin && x[i] < tmax)  
 {ypred[i] = exp(-a+ nn*log(x[i]-tmin)+mm*log(tmax-x[i])); 
    y[i] ~ normal(ypred[i], sigma);

}}
        tmin ~ gamma(.01,.01);

            nn ~ gamma(0.1, 0.1);
       mm ~ gamma(0.1, 0.1);

        tmax ~ gamma(.01,.001);

      a ~ gamma(0.1, 0.01);
sigmasq ~ inv_gamma(1e-3, 1e-3);}
  generated quantities { 
  real log_lik[N];
  real yeval[N];
  real dev;
  real tdmax;
  real scale[N];
  real alpha;
  int  count=0;
   real sscale2[NP];
   real mu[NP];
real ypred[NP];
  for (i in 1:N) 
  {
    if (x[i] > tmin && x[i] < tmax ) 
    {
    scale[i] = exp(-a+ nn*log(x[i]-tmin)+mm*log(tmax-x[i])); 
  log_lik[i] = normal_lpdf(y[i] | scale[i], sigma);
  yeval[i] = normal_rng(scale[i] , sigma);
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
if(is_nan(sscale2[j]))
{
  ypred[j]=0;
  sscale2[j]=0;
}else{
  ypred[j] = normal_rng(sscale2[j],sigma);
}
mu[j]=sscale2[j];}
  dev = -2*sum(log_lik[]);
  tdmax = (nn*tmax+mm*tmin)/(nn+mm);
    alpha= exp(-a);
}
