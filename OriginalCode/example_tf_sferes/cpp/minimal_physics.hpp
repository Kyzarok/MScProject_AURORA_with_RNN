#ifndef ___FIT_L0_HPP__
#define ___FIT_L0_HPP__


#define M 1.0
#define NB_STEP 50
#define DT 1.0
#define FMAX 200.0

FIT_QD(Minimal_physics)
{
public:
  Minimal_physics():_entropy(-1){  }

  
  typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;

  const std::vector<Eigen::VectorXf>& observations() const{
      return cart_traj;
  }

  template<typename block_t>
  void get_flat_observations(block_t& data)const
  {
    for(size_t t=0; t< observations().size(); t++)
      {
	data(0,t) = observations()[t][0];
	data(0,t+observations().size()) = observations()[t][1];
      }
  }
     
  size_t get_flat_obs_size()const{
    assert(observations().size());
    return observations().size()*observations()[0].size();
  }
  
  const std::vector<float>& gt() {return _gt;}
  float& entropy() {return _entropy;}
  
  template<typename Indiv>
    void eval(Indiv& ind)
  {
      
    theta = ind.data(0)*M_PI/2;
    F = ind.data(1)*FMAX;
    simulate(F,theta);
    this->_value = -1;
    _gt = desc_hardcoded();
  }
  
  void simulate(float F, float theta)
  {
    Eigen::Vector2f a;
    a(0)= F * cos(theta) / M;
    a(1)=(F * sin(theta)-9.81)/ M;
    if( F * sin(theta) <= 9.81*3 )
      {
	Eigen::Vector2f p;
	p(0)=0;
	p(1)=0;
	for(size_t t=0;t<NB_STEP;t++)
	  {
	    cart_traj.push_back(p);
	    polar_traj.push_back(p);
	  }
	this->_dead=true;
	return;
      }
    Eigen::Vector2f v;
    v(0)=0;
    v(1)=0;
    Eigen::Vector2f p;
    p(0)=0;
    p(1)=0;
    Eigen::Vector2f polar;
    polar(0)=0;
    polar(1)=0;
    
    cart_traj.push_back(p);
    polar_traj.push_back(polar);
    
    for(size_t t=0;t<NB_STEP-1;t++)
      {
	v=v+a*DT;
	p=p+v*DT;
	a(0)=0;
	a(1)=-9.81; 
	if(p(1)<=0)//contact with the ground
	  {
	    p(1)=0;
	    a(1)=-0.6*v(1); //dumping factor
	    v(1)=0;
	  }
	
	polar(0)=p.norm();
	polar(1)=atan2(p(1),p(0));

	cart_traj.push_back(p);
	polar_traj.push_back(polar);
	
      }

  }
  std::vector<float> desc_fulldata() const
  {
    Mat data(1,this->get_flat_obs_size());
    this->get_flat_observations(data);
    std::vector<float> res(this->get_flat_obs_size());

    for(size_t i=0;i<res.size();i++)
      res[i]=data(0,i);

    return res;
  }

  std::vector<float> desc_genotype() const
  {
    std::vector<float> res(2);
    res[0]=theta/(M_PI/2.0)*2-1;
    res[1]=F/200.0*2-1;
    return res;
  }
  
  std::vector<float> desc_hardcoded() const
  {
    std::vector<float> res(2);
    // This represents the tip of the bell curve
    
    float Vx = std::cos(theta) * F;
    float Vy = std::sin(theta) * F - 9.81;
    float Px = Vx/2.0;
    float Py = Vy/2.0;
    float tmax = (std::sin(theta) * F)/9.81 - 1;
    res[0] = (Vx * tmax + Px )/2000*2-1; // quick normalization
    res[1] = (-9.81* 0.5 * tmax*tmax + Vy * tmax + Py) / 2000 * 2 - 1; //quick normalization
    
    /*
    //polar representation of the hardcoded?
    Mat res_2 = Mat(data.rows(),2);
    res_2.col(0) = ((res.col(0).array().square()+res.col(1).array().square()).sqrt())/1.15;
    res_2.col(1) = res.col(1).binaryExpr(res.col(0), [] (float a, float b) { return std::atan2(a,b);} );
    res_2.col(1) =res_2.col(1)/M_PI*2;
    res = res_2;*/
    return res;
  }
  

  
private:
  std::vector<float> _gt;
  std::vector<Eigen::VectorXf> cart_traj;
  std::vector<Eigen::VectorXf> polar_traj;
  
  float theta;
  float F;

  float _entropy;
};



#endif
