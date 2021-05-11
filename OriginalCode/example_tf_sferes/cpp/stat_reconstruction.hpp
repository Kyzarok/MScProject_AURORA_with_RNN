#ifndef STAT_RECONSTRUCTION_HPP_
#define STAT_RECONSTRUCTION_HPP_

#include <numeric>
#include <sferes/stat/stat.hpp>

//#define MAP_WRITE_PARENTS

namespace sferes
{
  namespace stat
  {
    SFERES_STAT(Reconstruction, Stat)
    {
    public:
      template<typename E>
	void refresh(const E& ea)
      {

	if (ea.gen() % Params::pop::dump_period == 0)
	   _write_container(std::string("reconst_"), ea);

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

	Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > data;
	Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > desc;
	Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > recon;
	boost::fusion::at_c<0>(ea.fit_modifier()).get_data(ea.pop(),data);
	boost::fusion::at_c<0>(ea.fit_modifier()).get_descriptor(data,desc);
	boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(data,recon);
	
	ofs<<desc.transpose()<<std::endl<<data.transpose()<<std::endl<<recon.transpose()<<std::endl;
 

	
      }
    };

  }
}

#endif
