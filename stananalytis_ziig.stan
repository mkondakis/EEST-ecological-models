functions{
  real inv_logit_shift(real x, real r, real an){
 real val;
 real test=an*(x-r);
  val=exp(- log_sum_exp(0,test));
return val;                              
}
}
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
  vector[N] status;
   int<lower=0> NP; 
   vector[NP] xpred;
  }
  transformed data {
 real an=100;
real r=5e-3;  // cause 0.05 is the median response and 0.1 is the max response.
}
  parameters {
  real <lower=4> tmin;
  real  <lower=0> nn;
  real <lower=0> a;
  real  <lower=0> mm;
  real <lower=tmin> tmax;
  real <lower=2> shape;
  }
  transformed parameters{
  vector<lower=0, upper=1>[N] th;
 

for (i in 1:N) {
  
  if ((x[i] > tmin) && (x[i] < tmax)) th[i]=inv_logit_shift(exp(-a+nn*log(x[i]-tmin)+mm*log(tmax-x[i])),r,an); else th[i]=1;
}
}
  model {
  real ypred[N];
  real log_zero;
  real log_other;
  for (i in 1:N)
  {
    
      if (status[i]==0)
      {    log_zero=log(th[i]);
        target += log_zero; 
            ypred[i]=0;
      }else
      {
    log_other=log(1-th[i]);
    ypred[i] = exp(-a+nn*log(x[i]-tmin)+mm*log(tmax-x[i]));   
    if ((!is_nan(ypred[i])) && (ypred[i]>0))  {
    target += log_other +
    inv_gamma_lpdf(y[i] | shape , (shape-1)* ypred[i]);
    } else  ypred[i]=0
    ;
    }
}
   tmin ~ gamma(.01,.01);
    nn ~ gamma(0.1, 0.1);
       a ~ normal(0,100);
        tmax ~ gamma(.01,.001);
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
   real mu[NP];
real ypred[NP];
real th2[NP];

  count=0;
  for (i in 1:N) 
  {
    
        scale[i] = exp(-a+nn*log(x[i]-tmin)+mm*log(tmax-x[i]));
if (status[i]==0) 
    {
    if (!is_nan(th[i])) log_lik[i] = bernoulli_lpmf(1 | th[i]); else log_lik[i] = 0; 
    scale[i] = 0;
    }
    else if(!is_nan(scale[i]) && (!is_nan(th[i]))&&  scale[i]>0 && status[i]!=0)  
    {
    log_lik[i] = bernoulli_lpmf(0 | th[i])+inv_gamma_lpdf(y[i] | shape,(shape-1)*scale[i]); 
    } else
    {
    scale[i] = 0;  
    log_lik[i] =0;
    count +=1;
    }
    if(bernoulli_rng(th[i])|| scale[i]<=0 || is_nan(scale[i])) yeval[i] =0; else yeval[i]=inv_gamma_rng(shape , scale[i]);  
  }
  
  for (j in 1:NP) {
    
    if ((xpred[j] > tmin) && (xpred[j] < tmax)) th2[j]=inv_logit_shift(exp(-a+ nn*log(xpred[j]-tmin)+mm*log(tmax-xpred[j])),r,an); else th2[j]=1;
mu[j] = exp(-a+ nn*log(xpred[j]-tmin)+mm*log(tmax-xpred[j]));
if(bernoulli_rng(th2[j]) || mu[j]<=0 || is_nan(mu[j]))
{
  ypred[j]=0;
    mu[j] =0;
}else{
   ypred[j] =(inv_gamma_rng(shape,(shape-1)*mu[j]));

  }}
  alpha= exp(-a);
  dev = -2*sum(log_lik[]);
  tdmax = (nn*tmax+mm*tmin)/(nn+mm);
} 
