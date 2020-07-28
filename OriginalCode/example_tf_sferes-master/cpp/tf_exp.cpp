//| This file is a part of the sferes2 framework.
//| Copyright 2016, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#include <iostream>

#include <sferes/eval/parallel.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/run.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/qd_container.hpp>
#include <sferes/stat/qd_selection.hpp>
#include <sferes/stat/qd_progress.hpp>


#include <sferes/fit/fit_qd.hpp>
#include <sferes/qd/container/archive.hpp>
#include <sferes/qd/container/kdtree_storage.hpp>
#include <sferes/qd/quality_diversity.hpp>
#include <sferes/qd/selector/value_selector.hpp>
#include <sferes/qd/selector/score_proportionate.hpp>


#include "network_loader.hpp"
#include "minimal_physics.hpp"
#include "preprocessor.hpp"
#include "modifier_dim_red.hpp"


#include "stat_projection.hpp"
#include "stat_reconstruction.hpp"

typedef     Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor > Mat;



using namespace sferes::gen::evo_float;

struct Params {
  SFERES_CONST size_t update_period = 50;
  
  struct nov {
    static double l;
    SFERES_CONST double k = 15; 
    SFERES_CONST double eps = 0.1;
  };
  
  // TODO: move to a qd::
  struct pop {
        // size of a batch
    SFERES_CONST size_t size = 200;
    SFERES_CONST size_t nb_gen = 10001;
    SFERES_CONST size_t dump_period = 100;
  };
  struct parameters {
    // used to avoid really degenerated experiments with trajectories not doing anything.
    SFERES_CONST float min = 0.1;
    SFERES_CONST float max = 0.9;
  };
  struct evo_float {
    SFERES_CONST float cross_rate = 0.0f;
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float eta_m = 10.0f;
    SFERES_CONST float eta_c = 10.0f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
  };
  struct qd {
    SFERES_CONST size_t dim = 2;
    SFERES_CONST size_t behav_dim = 2;
  };
};

template<typename fit_t>
void gen_dataset(Mat& data)
{
  size_t reso=100;
  data=Mat(reso*reso, 100);
  
  size_t index=0;
  for(size_t i=0;i<reso;i++)
    for(size_t j=0;j<reso;j++)
      {
        fit_t fit;
        fit.simulate(i/(float)reso*200.0 , j/(float)reso*M_PI/2);
	
        for(size_t t=0; t<fit.observations().size(); t++)
          {
            data(index,t)=fit.observations()[t][0];
            data(index,t+50)=fit.observations()[t][1];
          }
        index++;
      }
}

double Params::nov::l;

// quick hack to have "write" access to the container, this need to be added to the main API later.
template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename Select, typename Container , typename Params, typename Exact = stc::Itself>
class QualityDiversity_2 : public sferes::qd::QualityDiversity < Phen, Eval, Stat, FitModifier, Select, Container, Params, typename stc::FindExact<QualityDiversity_2<Phen, Eval, Stat, FitModifier, Select, Container, Params, Exact>, Exact>::ret> 
{
  public:

  typedef Phen phen_t;
  typedef boost::shared_ptr<Phen> indiv_t;
  typedef typename std::vector<indiv_t> pop_t;
  typedef typename pop_t::iterator it_t;
  

  
  Container& container() { return this->_container; }

  void add(pop_t& pop_off, std::vector<bool>& added, pop_t& pop_parents){
    this->_add(pop_off, added, pop_parents);
  }
  
  // Same function, but without the need of parent.
  void add(pop_t& pop_off, std::vector<bool>& added ){
    std::cout<<"adding with l: "<<Params::nov::l<<std::endl;
    this->_add(pop_off, added);
  }

};

  

int main(int argc, char **argv) 
{
  srand(time(0));
  Params::nov::l = 0.01;


  using namespace sferes;
  typedef Minimal_physics<Params> fit_t;
  typedef gen::EvoFloat<2, Params> gen_t;
  typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
  
  typedef modif::ModifDimRed<phen_t,Params> modifier_t;
  
  typedef qd::container::KdtreeStorage< boost::shared_ptr<phen_t>, Params::qd::behav_dim > storage_t;
  typedef qd::container::Archive<phen_t, storage_t, Params> container_t;
  typedef qd::selector::ScoreProportionate<phen_t,qd::selector::getCuriosity, Params> select_t;

  typedef eval::Parallel<Params> eval_t;

  typedef boost::fusion::vector<
    stat::QdContainer<phen_t, Params>,
    stat::QdProgress<phen_t, Params>,
    stat::Projection<phen_t, Params>,
    stat::Reconstruction<phen_t,Params> > stat_t;
  
  
  typedef QualityDiversity_2<phen_t, eval_t, stat_t, modifier_t, select_t, container_t, Params> ea_t;

  ea_t qd;
  run_ea(argc, argv, qd);
  
    

}
