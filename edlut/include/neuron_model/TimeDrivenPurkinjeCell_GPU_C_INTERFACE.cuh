/***************************************************************************
 *                           TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cuh    *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Richard Carrill, Niceto Luque and    *
						  Francisco Naveros	   *
 * email                : rcarrillo@ugr.es, nluque@ugr.es and		   *
						  fnaveros@ugr.es    	   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENPURKINJECELL_GPU_C_INTERFACE_H_
#define TIMEDRIVENPURKINJECELL_GPU_C_INTERFACE_H_

/*!
 * \file TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cuh
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date December 2015
 *
 * This file declares a class which implements a Purkinje cell model. This is an Hodgkin and Huxley (HH)
 * neuron model with three differential equations and four time dependent equations (conductances and 
 * external current). This model is implemented in CPU to control a GPU class.
 */

#include "./TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh"

#include <string>



using namespace std;

class InputSpike;
class VectorNeuronState;
class VectorNeuronState_GPU_C_INTERFACE;
class Interconnection;

class TimeDrivenPurkinjeCell_GPU2;


/*!
 * \class TimeDrivenPurkinjeCell_GPU_C_INTERFACE
 *
 * \brief Time driven neuron model with a membrane potential, two current channels, three conductances
 * one external current. This model is implemented in CPU to control a GPU class..
 *
 * \author Richard Carrillo
 * \author Niceto Luque
 * \author Francisco Naveros
 * \date December 2015
 */
class TimeDrivenPurkinjeCell_GPU_C_INTERFACE : public TimeDrivenNeuronModel_GPU_C_INTERFACE {
	protected:
		/*!
		* \brief leak current in mS/cm^2 units
		*/
		const float g_leak;

		/*!
		* \brief high-threshold noninactivating calcium conducance in mS/cm^2 units
		*/
		const float g_Ca;

		/*!
		* \brief muscarinic receptor suppressed potassium conductance (or M conductance) in mS/cm^2 units
		*/
		const float g_M;

		/*!
		* \brief Cylinder length of the soma in cm units
		*/
		const float cylinder_length_of_the_soma;

		/*!
		* \brief Radius of the soma in cm units
		*/
		const float radius_of_the_soma;

		/*!
		* \brief Cell area in cm^2 units
		*/
		const float area;

		/*!
		* \brief Membrane capacitance in uF/cm^2 units
		*/
		const float c_m;
		
		/*!
		 * \brief Peak amplitude in mV units
		 */
		const float spk_peak;



		/*!
		* \brief Excitatory reversal potential in mV units
		*/
		float e_exc;

		/*!
		* \brief Inhibitory reversal potential in mV units
		*/
		float e_inh;

		/*!
		* \brief Firing threshold in mV units
		*/
		float v_thr;

		/*!
		* \brief Resting potential in mV units
		*/
		float e_leak;

		/*!
		* \brief AMPA receptor time constant in ms units
		*/
		float tau_exc;

		/*!
		* \brief GABA receptor time constant in ms units
		*/
		float tau_inh;
		
		/*!
		 * \brief NMDA receptor time constant in ms units
		 */
		float tau_nmda;

		/*!
		* \brief Refractory period in ms units
		*/
		float tau_ref;


		/*!
		 * \brief Neuron model in the GPU.
		*/
		TimeDrivenPurkinjeCell_GPU2 ** NeuronModel_GPU2;

		/*!
		* \brief It initializes the CurrentSynapsis object.
		*
		* It initializes the CurrentSynapsis object.
		*/
		virtual void InitializeCurrentSynapsis(int N_neurons);

	public:


		/*!
		 * \brief Number of state variables for each cell.
		*/
		const int N_NeuronStateVariables=7;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell (V, Ca and M).
		*/
		const int N_DifferentialNeuronState=3;
			//Index of each differential neuron state variable
			const int V_m_index = 0;
			const int Ca_index = 1;
			const int M_index = 2;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell (EXC, INH, NMDA, EXT_I).
		 */
		const int N_TimeDependentNeuronState = 4;
			//Index of each time dependent neuron state variable
			const int EXC_index = N_DifferentialNeuronState;
			const int INH_index = N_DifferentialNeuronState + 1;
			const int NMDA_index = N_DifferentialNeuronState + 2;
			const int EXT_I_index = N_DifferentialNeuronState + 3;

		
		/*!
		 * \brief Boolean variable setting in runtime if the neuron model receive each one of the supported input synpase types (N_TimeDependentNeuronState) 
		 */
		bool EXC, INH, NMDA, EXT_I;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		TimeDrivenPurkinjeCell_GPU_C_INTERFACE();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenPurkinjeCell_GPU_C_INTERFACE();

		/*!
		 * \brief It return the Neuron Model VectorNeuronState 
		 *
		 * It return the Neuron Model VectorNeuronState 
		 *
		 */
		virtual VectorNeuronState * InitializeState();

		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param inter the interconection which propagate the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time);

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		virtual void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current);

		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, double CurrentTime);

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		enum NeuronModelOutputActivityType GetModelOutputActivityType();

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		enum NeuronModelInputActivityType GetModelInputActivityType();

		/*!
		 * \brief It prints the time-driven model info.
		 *
		 * It prints the current time-driven model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);


		/*!
		 * \brief It initialice a neuron model in GPU.
		 *
		 * It initialice a neuron model in GPU.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeClassGPU2(int N_neurons);

		/*!
		 * \brief It delete a neuron model in GPU.
		 *
		 * It delete a neuron model in GPU.
		 */
		virtual void DeleteClassGPU2();

		/*!
		 * \brief It create a object of type VectorNeuronState_GPU2 in GPU.
		 *
		 * It create a object of type VectorNeuronState_GPU2 in GPU.
		 */
		virtual void InitializeVectorNeuronState_GPU2();


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection);

		/*!
		* \brief It returns the neuron model parameters.
		*
		* It returns the neuron model parameters.
		*
		* \returns A dictionary with the neuron model parameters
		*/
		virtual std::map<std::string, boost::any> GetParameters() const;

		/*!
		* \brief It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* It returns the neuron model parameters for a specific neuron once the neuron model has been initilized with the number of neurons.
		*
		* \param index neuron index inside the neuron model.
		*
		* \returns A dictionary with the neuron model parameters
		*
		* NOTE: this function is accesible throgh the Simulatiob_API interface.
		*/
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const ;

		/*!
		* \brief It loads the neuron model properties.
		*
		* It loads the neuron model properties from parameter map.
		*
		* \param param_map The dictionary with the neuron model parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*/
		virtual void SetParameters(std::map<std::string, boost::any> param_map) ;

		/*!
		* \brief It creates the integration method
		*
		* It creates the integration methods using the parameter map.
		*
		* \param param_map The dictionary with the integration method parameters.
		*
		* \throw EDLUTException If it happens a mistake with the parameters in the dictionary.
		*/
		virtual IntegrationMethod_GPU_C_INTERFACE * CreateIntegrationMethod(ModelDescription imethodDescription) ;

		/*!
		* \brief It returns the default parameters of the neuron model.
		*
		* It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		* set for this neuron model.
		*
		* \returns A dictionary with the neuron model default parameters.
		*/
		static std::map<std::string, boost::any> GetDefaultParameters();

		/*!
		* \brief It creates a new neuron model object of this type.
		*
		* It creates a new neuron model object of this type.
		*
		* \param param_map The neuron model description object.
		*
		* \return A newly created InputNeuronModel object.
		*/
		static NeuronModel* CreateNeuronModel(ModelDescription nmDescription);

		/*!
		* \brief It loads the neuron model description and tables (if necessary).
		*
		* It loads the neuron model description and tables (if necessary).
		*
		* \param FileName This parameter is not used. It is stub parameter for homegeneity with other neuron models.
		*
		* \return A neuron model description object with the parameters of the neuron model.
		*/
		static ModelDescription ParseNeuronModel(std::string FileName) ;

		/*!
		* \brief It returns the name of the neuron type
		*
		* It returns the name of the neuron type.
		*/
		static std::string GetName();

		/*!
		* \brief It returns the neuron model information, including its parameters.
		*
		* It returns the neuron model information, including its parameters.
		*
		*\return a map with the neuron model information, including its parameters.
		*/
		static std::map<std::string, std::string> GetNeuronModelInfo();

		/*!
		* \brief Comparison operator between neuron models.
		*
		* It compares two neuron models.
		*
		* \return True if the neuron models are of the same type and with the same parameters.
		*/
		virtual bool compare(const NeuronModel * rhs) const{
			if (!TimeDrivenNeuronModel_GPU_C_INTERFACE::compare(rhs)){
				return false;
			}
			const TimeDrivenPurkinjeCell_GPU_C_INTERFACE * e = dynamic_cast<const TimeDrivenPurkinjeCell_GPU_C_INTERFACE *> (rhs);
			if (e == 0) return false;

			return this->e_exc==e->e_exc &&
				this->e_inh==e->e_inh &&
				this->v_thr==e->v_thr &&
				this->e_leak==e->e_leak &&
				this->tau_exc==e->tau_exc &&
				this->tau_inh==e->tau_inh &&
				this->tau_nmda==e->tau_nmda &&
				this->tau_ref==e->tau_ref;
        };

};

#endif /* TIMEDRIVENPURKINJECELL_GPU_C_INTERFACE_H */
