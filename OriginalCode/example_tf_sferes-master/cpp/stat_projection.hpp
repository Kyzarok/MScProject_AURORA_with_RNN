#ifndef STAT_PROJECTION_HPP_
#define STAT_PROJECTION_HPP_

#include <numeric>
#include <sferes/stat/stat.hpp>

typedef     Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;

//#define MAP_WRITE_PARENTS

namespace sferes
{
  namespace stat
  {
    SFERES_STAT(Projection, Stat)
    {
    public:
      template<typename E>
	void refresh(const E& ea)
      {

	if (ea.gen() % Params::pop::dump_period == 0)
	   _write_container(std::string("proj_"), ea);

      }
      template<typename EA>
	void _write_container(const std::string& prefix,
			    const EA& ea) const
      {
        std::cout << "writing..." << prefix << ea.gen() << std::endl;
        std::string fname = ea.res_dir() + "/"
	  + prefix
	  + boost::lexical_cast<
	std::string>(ea.gen())
	  + std::string(".dat");
	
        std::ofstream ofs(fname.c_str());
	
        size_t offset = 0;
	ofs.precision(17);
        for(auto it = ea.pop().begin(); it != ea.pop().end(); ++it)
	  {
	    ofs << offset << "    "<<(*it)->fit().entropy()<<"    ";
	    
	    for(size_t dim = 0; dim < (*it)->fit().desc().size(); ++dim)
	      ofs << (*it)->fit().desc()[dim] << " ";
	    //ofs << " " << array(idx)->fit().value() << std::endl;
	    ofs<<"     ";
	    for(size_t dim = 0; dim < (*it)->fit().gt().size(); ++dim)
	      ofs << (*it)->fit().gt()[dim] << " ";

	    
	    Mat traj(1, (*it)->fit().get_flat_obs_size());
	    (*it)->fit().get_flat_observations(traj);
	    ofs<<traj<<"   ";
	    
	    ofs<<std::endl;
            ++offset;
	  }
      }

    };

  }
}

#endif
