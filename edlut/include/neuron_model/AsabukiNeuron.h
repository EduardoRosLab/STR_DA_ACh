/***************************************************************************
 *                           AsabukiNeuron.h                               *
 *                           ---------------                               *
 * copyright            : (C) 2022 by Álvaro González-Redondo              *
 * email                : alvarogr@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef ASABUKINEURON_H_
#define ASABUKINEURON_H_

/*!
 * \file AsabukiNeuron.h
 *
 * \author Álvaro González-Redondo
 * \date June 2022
 *
 * This file declares a class which implements a neuron model that can learn as
 * in Asabuki et al. (2020). It uses the synaptic model called AsabukiSynapse.
 */

#include "neuron_model/TimeDrivenNeuronModel.h"

#include <string>

using namespace std;

class VectorNeuronState;
class InternalSpike;
class Interconnection;
struct ModelDescription;

/*!
 * \class AsabukiNeuron
 *
 * \brief This file declares a class which implements a neuron model that can
 * learn as in Asabuki et al. (2020). It uses the synaptic model called
 * AsabukiSynapse.
 *
 * \author Álvaro González-Redondo
 * \date June 2022
 */
class AsabukiNeuron : public TimeDrivenNeuronModel
{
protected:

	float tau_exc, inv_tau_exc;
	float tau_inh, inv_tau_inh;
	float tau_nmda, inv_tau_nmda;

	float tau, g_l;
	float tau_ref;
	float g_d, alpha;

	float t_0, inv_t_0;
	float beta_0, theta_0, exp_bt, phi_0;

	/*!
	 * \brief It computest the g_nmda_inf values based on the e_exc and e_inh values.
	 *
	 * It computest the g_nmda_inf values based on the e_exc and e_inh values.
	 */
	virtual void Generate_g_nmda_inf_values();

	/*!
	 * \brief It returns the g_nmda_value corresponding with the membrane potential (V_m).
	 *
	 * It returns the g_nmda_value corresponding with the membrane potential (V_m).
	 *
	 * \param V_m membrane potential.
	 *
	 * \return g_nmda_value corresponding with the membrane potential.
	 */
	virtual float Get_g_nmda_inf(float V_m);

	/*!
	 * \brief It initializes the CurrentSynapsis object.
	 *
	 * It initializes the CurrentSynapsis object.
	 */
	virtual void InitializeCurrentSynapsis(int N_neurons);


	static const int TableSizeExp = 256;
	float exp_x_min = -60.0f;
	float exp_x_max = 60.0f;
	float texp[TableSizeExp];

	void GenerateTableExp();
	inline float get_exp(float x);


public:
	/*!
	 * \brief Number of state variables for each cell.
	 */
	const int N_NeuronStateVariables = 10;

	/*!
	 * \brief Number of state variables which are calculate with a differential equation for each cell.
	 */
	const int N_DifferentialNeuronState = 6;
	// Index of each differential neuron state variable
	const int n_index = 0;
	const int u_index = 1;
	const int v_index = 2;
	const int mean_index = 3; // used for online u mean calculation
	const int M2_index = 4;	  // used for online u mean calculation
	const int V_m_index = 5;

	/*!
	 * \brief Number of state variables which are calculate with a time dependent equation for each cell (EXC, INH, NMDA, EXT_I).
	 */
	const int N_TimeDependentNeuronState = 4;
	// Index of each time dependent neuron state variable
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
	AsabukiNeuron();

	/*!
	 * \brief Class destructor.
	 *
	 * It destroys an object of this class.
	 */
	virtual ~AsabukiNeuron();

	/*!
	 * \brief It return the Neuron Model VectorNeuronState
	 *
	 * It return the Neuron Model VectorNeuronState
	 *
	 */
	virtual VectorNeuronState *InitializeState();

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
	virtual InternalSpike *ProcessInputSpike(Interconnection *inter, double time);

	/*!
	 * \brief It processes a propagated current (input current in the cell).
	 *
	 * It processes a propagated current (input current in the cell).
	 *
	 * \param inter the interconection which propagate the spike
	 * \param target the neuron which receives the spike
	 * \param Current input current.
	 */
	virtual void ProcessInputCurrent(Interconnection *inter, Neuron *target, float current);

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
	virtual ostream &PrintInfo(ostream &out);

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
	 * \brief It evaluates if a neuron must spike.
	 *
	 * It evaluates if a neuron must spike.
	 *
	 * \param previous_V previous membrane potential
	 * \param NeuronState neuron state variables.
	 * \param index Neuron index inside the neuron model.
	 * \param elapsedTimeInNeuronModelScale integration method step.
	 * \return It returns if a neuron must spike.
	 */
	void EvaluateSpikeCondition(float previous_V, float *NeuronState, int index, float elapsedTimeInNeuronModelScale);

	/*!
	 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
	 *
	 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
	 *
	 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
	 * \param AuxNeuronState results of the differential equations evaluation.
	 * \param index Neuron index inside the VectorNeuronState
	 */
	void EvaluateDifferentialEquation(float *NeuronState, float *AuxNeuronState, int index, float elapsed_time);

	/*!
	 * \brief It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
	 *
	 * It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
	 *
	 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
	 * \param elapsed_time integration time step.
	 * \param elapsed_time_index index inside the conductance_exp_values array.
	 */
	void EvaluateTimeDependentEquation(float *NeuronState, int index, int elapsed_time_index);

	/*!
	 * \brief It Checks if the neuron model has this connection type.
	 *
	 * It Checks if the neuron model has this connection type.
	 *
	 * \param Conncetion input connection type.
	 *
	 * \return If the neuron model supports this connection type
	 */
	virtual bool CheckSynapseType(Interconnection *connection);

	/*!
	 * \brief It calculates the conductace exponential value for an elapsed time.
	 *
	 * It calculates the conductace exponential value for an elapsed time.
	 *
	 * \param index elapsed time index .
	 * \param elapses_time elapsed time.
	 */
	void Calculate_conductance_exp_values(int index, float elapsed_time);

	/*!
	 * \brief It calculates the number of electrical coupling synapses.
	 *
	 * It calculates the number for electrical coupling synapses.
	 *
	 * \param inter synapse that arrive to a neuron.
	 */
	void CalculateElectricalCouplingSynapseNumber(Interconnection *inter);

	/*!
	 * \brief It allocate memory for electrical coupling synapse dependencies.
	 *
	 * It allocate memory for electrical coupling synapse dependencies.
	 */
	void InitializeElectricalCouplingSynapseDependencies();

	/*!
	 * \brief It calculates the dependencies for electrical coupling synapses.
	 *
	 * It calculates the dependencies for electrical coupling synapses.
	 *
	 * \param inter synapse that arrive to a neuron.
	 */
	void CalculateElectricalCouplingSynapseDependencies(Interconnection *inter);

	/*!
	 * \brief It gets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
	 *
	 * It gets the required parameter in the adaptative integration methods (Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF1 and Bifixed_BDF2).
	 *
	 * \param startVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
	 *  step from elapsedTimeInNeuronModelScale to bifixedElapsedTimeInNeuronModelScale.
	 * \param endVoltageThreshold, when the membrane potential reaches this value, the multi-step integration methods change the integration
	 *  step from bifixedElapsedTimeInNeuronModelScale to ElapsedTimeInNeuronModelScale after timeAfterEndVoltageThreshold in seconds.
	 * \param timeAfterEndVoltageThreshold, time in seconds that the multi-step integration methods maintain the bifixedElapsedTimeInNeuronModelScale
	 *  after the endVoltageThreshold
	 */
	virtual void GetBifixedStepParameters(float &startVoltageThreshold, float &endVoltageThreshold, float &timeAfterEndVoltageThreshold);

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
	virtual IntegrationMethod *CreateIntegrationMethod(ModelDescription imethodDescription) ;

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
	 * \return A newly created NeuronModel object.
	 */
	static NeuronModel *CreateNeuronModel(ModelDescription nmDescription);

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
	virtual bool compare(const NeuronModel *rhs) const
	{
		if (!TimeDrivenNeuronModel::compare(rhs))
		{
			return false;
		}
		const AsabukiNeuron *e = dynamic_cast<const AsabukiNeuron *>(rhs);
		if (e == 0)
			return false;

		// return this->e_exc==e->e_exc &&
		//        this->e_inh==e->e_inh &&
		//        this->e_leak==e->e_leak &&
		//        this->v_thr==e->v_thr &&
		//        this->c_m==e->c_m &&
		//        this->tau_exc==e->tau_exc &&
		//        this->tau_inh==e->tau_inh &&
		//        this->tau_ref==e->tau_ref &&
		//        this->g_leak == e->g_leak &&
		//        this->tau_nmda==e->tau_nmda;
		return true;
	};
};

#endif /* ASABUKINEURON_H_ */
