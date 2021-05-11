#ifndef __DIM_RED_MODIFIER_HPP__
#define __DIM_RED_MODIFIER_HPP__


#include <sferes/stc.hpp>
#include "preprocessor.hpp"

namespace sferes {
  namespace modif {
    template<typename Phen, typename Params>
    class ModifDimRed
    {
    public:
      typedef Phen phen_t;
      typedef boost::shared_ptr<Phen> indiv_t;
      typedef typename std::vector<indiv_t> pop_t;
      
      ModifDimRed():last_update(0),update_id(0){
	_prep.init();
	network = std::unique_ptr<NetworkLoader>(new NetworkLoader("exp/example_tf_sferes/resources/model_init.ckpt"));
      }

      typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;
      
      template<typename Ea>
      void apply(Ea& ea)
      {
	if (Params::update_period >0 && (ea.gen() == 1 || ea.gen() == last_update + Params::update_period * std::pow(2,update_id-1)))
	  { 
	    update_id++;
	    last_update=ea.gen();
	    update_descriptors(ea);
	  }

	if(!ea.offspring().size())
	  return;
	
	assign_desc(ea.offspring());
      }

      template<typename Ea>
      double get_avg_recon_loss(Ea& ea){
	Mat data;	
	this->collect_dataset(data, ea);
	float avg_recon = network_eval_data(data);
	return avg_recon;
      }
      
      void assign_desc(pop_t& pop)
      {
	pop_t filtered_pop;
	for(auto ind:pop)
	  if(!ind->fit().dead())
	    filtered_pop.push_back(ind);
	  else
	    {
	      std::vector<double>dd={-1,-1};
	      ind->fit().set_desc(dd);
	    }

	Mat data;	
	this->get_data(filtered_pop, data);
	Mat res;//will be resized


	get_descriptor(data,res);
	
	for(size_t i=0; i<filtered_pop.size();i++){
	  std::vector<double>dd={(double)res(i,0), (double)res(i,1)};
	  filtered_pop[i]->fit().set_desc(dd);
	  filtered_pop[i]->fit().entropy() = (float)res(i,2);
	}
      }
      
      void get_descriptor(const Mat & data, Mat& res)const
      {
	desc_ae(data, res);
      }

      
      template<typename EA>
      void update_descriptors(EA& ea)
      {
	Mat data;
	collect_dataset(data, ea); // gather the data from the indiv in the archive into a dataset
	train_network(data);
	update_container(ea);  // clear the archive and re-fill it using the new network
      }


      
      
      template <typename EA>
      void collect_dataset(Mat& data, EA& ea)
      {
	std::vector<typename EA::indiv_t> content;
	ea.container().get_full_content(content);
	size_t pop_size=content.size();
	
	get_data(content, data);
	std::cout<<"training set is composed of "<<data.rows()<<" samples  ("<<ea.gen()<<" archive size : "<<pop_size<<")"<<std::endl;	
      }
      


      
      
      void desc_ae(const Mat & data, Mat& res) const
      {
	Mat scaled_data;
	_prep.apply(data,scaled_data);
	Mat desc, entropy, loss, reconst;
	network->eval(scaled_data, desc, entropy, loss, reconst) ;
	res=Mat(desc.rows(),desc.cols()+entropy.cols());
	res<<desc,entropy;
      }
      
      void train_network(const Mat & data)
      {
	// we change the data normalisation each time we train/refine network, could cause small changes in loss between two trainings.
	_prep.init(data);
	Mat scaled_data;
	_prep.apply(data,scaled_data);
	float final_entropy = network->training(scaled_data);
      }

      void get_reconstruction(const Mat & data, Mat & res)const
      {
	 Mat scaled_data;
        _prep.apply(data,scaled_data);
	network->get_reconstruction(scaled_data,res);
      }
      
      
      double network_eval_data(const Mat & data, std::ostream& oo = std::cout) const
      {
	Mat scaled_data;
	_prep.apply(data,scaled_data);
	return network->get_avg_recon_loss(scaled_data);
      }

      void get_bd(const pop_t& pop, Mat& data)const
      {
	data=Mat(pop.size(), Params::qd::dim );

	for(size_t i=0; i<pop.size();i++)
	  {
	    auto desc = pop[i]->fit().desc();
	    for(size_t id=0;id<Params::qd::dim;id++)
	      data(i,id)=desc[id];
	  }
	
      }
      
      void get_data(const pop_t& pop, Mat& data)const
      {
	if(pop[0]->fit().dead())
	  std::cout<<"WTF?"<<std::endl;
	
	data=Mat(pop.size(), pop[0]->fit().get_flat_obs_size() );

	for(size_t i=0; i<pop.size();i++)
	  {
	    auto row=data.row(i);
	    pop[i]->fit().get_flat_observations( row );
	  }
      }


      

      void distance(const Mat& X, Mat& dist)const{
	//std::cout<<"Neg distance"<<std::endl;
	// Compute norms
	Mat XX = X.array().square().rowwise().sum();
	Mat XY = (2*X)*X.transpose();
	
	// Compute final expression
	dist = XX * Eigen::MatrixXf::Ones(1,XX.rows());
	dist = dist + Eigen::MatrixXf::Ones(XX.rows(),1) * (XX.transpose());
	dist = dist - XY;
	//std::cout<<"END Neg distance"<<std::endl;
      }


      float get_new_l(const pop_t& pop)const{
	Mat data;
	this->get_bd(pop, data);
	Mat dist;
	this->distance(data, dist);
	float maxdist=sqrt(dist.maxCoeff());
	float K=60000; // arbitrary value to have a specific "resolution"
	return maxdist/sqrt(K);
      }

  
      
            
      template<typename Ea>
      void update_container(Ea& ea)
      {
	pop_t tmp;
	// Copy of the containt of the container into the _pop object.
	ea.container().get_full_content(tmp);
	ea.container().erase_content();
	std::cout<<"size pop: "<<tmp.size()<<std::endl;

	auto stat1 = get_stat(tmp);
	this->assign_desc(tmp);
	auto stat2= get_stat(tmp);

	// update l to maintain a number of indiv lower than 10k 
	Params::nov::l = this->get_new_l(tmp);
	std::cout<<"NEW L= "<<Params::nov::l<<std::endl;

	
	// Addition of the offspring to the container
	std::vector<bool> added;
	ea.add(tmp, added);
	ea.pop().clear();
	// Copy of the containt of the container into the _pop object.
	ea.container().get_full_content(ea.pop());

	dump_data(ea,stat1,stat2,added);

	ea.pop().clear();
	// Copy of the containt of the container into the _pop object.
	ea.container().get_full_content(ea.pop());
	std::cout<<ea.gen()<<" size pop2: "<<ea.pop().size()<<std::endl;
      }

      std::vector<std::pair<std::vector<double>,float> > get_stat(const pop_t& pop){
	std::vector<std::pair<std::vector<double>,float> > result;
	for(auto ind:pop)
	  result.push_back({ind->fit().desc(),ind->fit().value()});
	return result;
      }
      
      template <typename EA>
      void dump_data( const EA& ea,
		      const std::vector<std::pair<std::vector<double>,float> >& stat1,
		      const std::vector<std::pair<std::vector<double>,float> >& stat2,
		      const std::vector<bool>& added ) const
      {
	std::cout << "writing... update_archive" << ea.gen() << std::endl;
	std::string fname = ea.res_dir() + "/"
	  + "update_archive"
	  + boost::lexical_cast<
	    std::string>(ea.gen())
	  + std::string(".dat");

	std::ofstream ofs(fname.c_str());

	assert(stat1.size()==stat2.size());
	assert(stat1.size()==added.size());
	
	for(size_t i=0;i<stat1.size();i++){
	  ofs<<i<<"  ";
	  for(size_t j=0;j<stat1[i].first.size();j++)
	    ofs<<stat1[i].first[j]<<" ";
	  ofs<<stat1[i].second<<"     ";
	  for(size_t j=0;j<stat2[i].first.size();j++)
	    ofs<<stat2[i].first[j]<<" ";
	  ofs<<stat2[i].second<<"             ";
	  ofs<<added[i]<<std::endl;
	}
      }

      const std::unique_ptr<NetworkLoader>& get_network()const {return network;}
      
      std::unique_ptr<NetworkLoader> network;
      Rescale_feature _prep;
      size_t last_update;
      size_t update_id;

      
    };
  }
}



#endif
