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
 val =a*(x-tmin)-pow(b,x)/pow(b,tmax);
 y_x[1] = val;
return y_x;                              
}
  }
data {
  int<lower=0> N;
  int<lower=0> NP; 
vector[NP] xpred;
vector[N] y;
vector[N] x;
}
transformed data{
vector[N] ey=exp(y);
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
real <lower=0> tmax;
real <lower=0, upper=1>  a;
real <lower=1>  b;
real <lower=0, upper=tmax> tmin;
real <lower=2> shape;
}
transformed parameters{
 vector[4] theta;
 theta[1]=a;
 theta[2]=b;
 theta[3]=tmin;
 theta[4]=tmax;
}
model {
vector[N]  mu;
for (i in 1:N)
{
      mu[i] = a*(x[i])-a*tmin-pow(b,x[i]-tmax);

  mu[i] = (shape-1)*mu[i];
  y[i] ~ inv_gamma(shape, mu[i]);
}

tmax ~ gamma(.1, .01); 
tmin ~ gamma(.1, .01); 
shape ~ gamma(.1, .01); 
a ~ beta(1,1);
b ~ gamma(.2, .1);
}
generated quantities { 
real log_lik[N];
real yeval[N];
real sscale[N];
real dev;
real  <lower=tmin, upper=tmax> tdmax;
vector[1] xmin;
vector[1] xmax;
real ypred[NP];
real sscale2[NP];
real mu[NP];

real s2cale=0;
xmin = algebra_solver(fbieri, y_guess1, theta, x_r, x_i, 1e-10, 1e+1, 1e+10);
xmax = algebra_solver(fbieri, y_guess2, theta, x_r, x_i, 1e-10, 1e+1, 1e+10);


for (i in 1:N) {

sscale[i] = (a*(x[i])-a*tmin-pow(b,x[i]-tmax));
  if(!is_inf(sscale[i]) && !is_nan(sscale[i]) &&  sscale[i]>0 )
{
  sscale[i] = (shape-1)*(sscale[i]);
  log_lik[i] = inv_gamma_lpdf(y[i] |shape , sscale[i]);
  yeval[i] = inv_gamma_rng(shape,sscale[i]);
}else{
  log_lik[i] = 0;
  yeval[i] = 0;
}
}
for (j in 1:NP) {
sscale2[j] =(a*(xpred[j])-a*tmin-pow(b,xpred[j]-tmax)); 
  if(!is_inf(sscale2[j]) && !is_nan(sscale2[j]) &&  sscale2[j]>0 )
{
  sscale2[j] = (shape-1)*(sscale2[j]);
  ypred[j] =log(inv_gamma_rng(shape,sscale2[j]));
  mu[j] = (a*(xpred[j])-a*tmin-pow(b,xpred[j]-tmax)); 

}else{
  ypred[j]=0;
  sscale2[j] =0;
  mu[j] = 0;
 }}

dev = -2*sum(log_lik[]);
tdmax = (log(a)-log(log(b)))/log(b)+tmax;
} 
