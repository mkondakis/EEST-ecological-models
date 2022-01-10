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
    //rat = shape * log(shape-1) - shape*N*l + shape*sum(exp(x*ro)) - shape*a*sum(exp(x*del));
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
  } 
 real inv_logit_shift(real x, real r, real an){
 real val;
 real test=an*(x-r);
  //val = 1/(1+exp(10*(x+r))); //to be  100 times faster and more sensitive
  //if (test> 700) val = 0; else if (test< - 700) val = 1;  else val=exp(- log_sum_exp(0,test));
  //if (test< - 700) val = 1;  else 
  val=exp(- log_sum_exp(0,test));
  //val =1-inv_logit(an*(x+r)); //to be  100 times faster and more sensitive
//if (x> 7) val = 0; else if (x<0) val = 1;  else val=exp(- 100*(x));
//if (test> 700) val = 0; else if ((test<= 0)&&(r<0)) val = 1;  else val=exp(- 20*(x-r));

return val;                              
}
//log(exp(10*x-10*max(x)))+10*max(x)
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
//vector[N] ey=exp(y);
real an=100;
//real<lower=0> an=100;
real r=5e-3; //r=3.006212e-04; // cause 0.05 is the median response and 0.1 is the max response.
//real r=-0.001; // cause 0.05 is the median response and 0.1 is the max response.
//real r=0; // cause 0.05 is the median response and 0.1 is the max response.
real l1=min(x)-min(x)/2;
real l2=max(x)+max(x)/2;
vector[1] y_guess1;
vector[1] y_guess2;
real x_r[0];
int x_i[0];
y_guess1[1]= l1;
y_guess2[1]= l2;
}
parameters {
 real <lower=0, upper=1> del; //<lower=0>
  real <lower=0, upper=del>  ro; // free in previous case
   real  <lower=((log(ro)-log(del))/(ro-del))> tmax;
  //real <lower=0, upper=(ro/del)> a;
  real  l; 
  real <lower=2> shape;

//real<lower=0, upper=1> th[N];

//real <lower=0> st;
}
transformed parameters{
  real <lower=0, upper=(ro/del)> a;
  vector<lower=0, upper=1>[N] th;
 vector[4] theta;
 a=exp((ro-del)*tmax);
 theta[1]=l;
 theta[2]=ro;
 theta[3]=a;
 theta[4]=del;


for (i in 1:N) {
   th[i]=inv_logit_shift((-l+exp(ro*x[i])-a*exp(del*x[i])),r,an);
}
}
model {
real log_other;
real log_zero;
vector[N]  ypred;

for (i in 1:N)
{
  
  if   (status[i]==0)   {
        ypred[i]=0;
        log_zero=log(th[i]);
       target += log_zero;
} else{
    ypred[i] = (-l+exp(ro*x[i])-a*exp(del*x[i]));
      log_other=log(1-th[i]);
      target += log_other +Sin_lpdf(y[i]|x[i], shape, ypred[i], a, l,ro , del); 
     
} 
}

target += gamma_lpdf(tmax|0.1, 0.01);
target += beta_lpdf(del|1, 1);
target += normal_lpdf(l|0, 1);
target += beta_lpdf(ro|1, 1);
target += gamma_lpdf(shape|0.1, 0.01);

}
generated quantities { 
real log_lik[N];
real yeval[N];
real scale[N];
real dev;
real tdmax;
vector[1] xmin;
vector[1] xmax;
real ypred[NP];
real sscale2[NP];
real th2[NP];
real mu[NP];

real s2cale=0;
xmin = algebra_solver(flactin, y_guess1, theta, x_r, x_i, 1e-10, 1e+1, 1e+10);
xmax = algebra_solver(flactin, y_guess2, theta, x_r, x_i, 1e-10, 1e+1, 1e+10);

for (i in 1:N) {
scale[i] =  (-l+exp(ro*x[i])-a*exp(del*x[i]));
 if (status[i]==0) 
    {
    if (!is_nan(th[i])) log_lik[i] = bernoulli_lpmf(1 | th[i]); else log_lik[i] = 0; 
    }
    else if(!is_nan(scale[i]) && (!is_nan(th[i])) &&  scale[i]>0 && status[i]!=0)  
    {
    log_lik[i] = bernoulli_lpmf(0 | th[i])+Sin_lpdf(y[i]|x[i], shape, scale[i], a, l,ro , del); 
    } else
    {
    scale[i] = 0;  
    log_lik[i] =0;
    }
    if(bernoulli_rng(th[i])|| scale[i]<=0 || is_nan(scale[i])) yeval[i] =0; else yeval[i]=inv_gamma_rng(shape , scale[i]);  
  }
for (j in 1:NP) {
th2[j]=inv_logit_shift((-l+exp(ro*xpred[j])-a*exp(del*xpred[j])),r,an);
sscale2[j] =(-l+exp(ro*xpred[j])-a*exp(del*xpred[j]));; 

if(bernoulli_rng(th2[j]) || sscale2[j]<=0 || is_nan(sscale2[j]))
{
  ypred[j]=0;
  sscale2[j] =0;
    mu[j] =0;
}else{
   ypred[j] =(inv_gamma_rng(shape,(shape-1)*sscale2[j]));
  mu[j] = sscale2[j]; 

  }}
dev = -2*sum(log_lik[]);
tdmax = tmax-log(ro/del)/(ro-del);
}
