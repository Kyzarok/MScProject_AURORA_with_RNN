#ifndef __PREPROCESSOR__HPP__
#define __PREPROCESSOR__HPP__


class Rescale_feature{

public:

  Rescale_feature():no_prep(true){}

  typedef     Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;

  void init()
  {
    no_prep=true;
  }
  void init(const Mat & data)
  {
    no_prep=false;
    //_max = data.colwise().maxCoeff();
    //_min = data.colwise().minCoeff();

    auto max = data.colwise().maxCoeff();
    auto min = data.colwise().minCoeff();
    auto max_1 = data.leftCols(max.size()/2).maxCoeff();
    auto max_2 = data.rightCols(max.size()/2).maxCoeff();
    _max=Eigen::VectorXf(max.size());
    _max.head(max.size()/2)=Eigen::VectorXf::Ones(max.size()/2)*max_1;
    _max.tail(max.size()/2)=Eigen::VectorXf::Ones(max.size()/2)*max_2;


    auto min_1 = data.leftCols(min.size()/2).minCoeff();
    auto min_2 = data.rightCols(min.size()/2).minCoeff();
    _min=Eigen::VectorXf(min.size());
    _min.head(max.size()/2)=Eigen::VectorXf::Ones(min.size()/2)*min_1;
    _min.tail(max.size()/2)=Eigen::VectorXf::Ones(min.size()/2)*min_2;

  }

  void apply(const Mat & data, Mat& res)const 
  {
    if(no_prep)
      {
      res=data;
      }
    else
      {
	res = (data.rowwise()-_min.transpose()).array().rowwise()/(_max-_min).transpose().array();
	res = (res.array() == res.array()).select(res, 0);
	res = (res*2).array()-1;
      }
  }
  
private: 
  Eigen::VectorXf _min;
  Eigen::VectorXf _max;
  bool no_prep;
};

#endif
