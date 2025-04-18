/***************************************************************************
 *                           LIFTimeDrivenModel_1_4_NEW.h                  *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef LIFTIMEDRIVENMODEL_1_4_NEW_H_
#define LIFTIMEDRIVENMODEL_1_4_NEW_H_

/*!
 * \file LIFTimeDrivenModel.h
 *
 * \author Francisco Naveros
 * \date October 2016
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model with one 
 * differential equation and four time dependent equations (conductances). This class is a
 * redefinition of LIFTimeDrivenModel_1_4 neuron model using mV, ms, nS and pF units.
 */

#include "./TimeDrivenNeuronModel.h"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;
class Interconnection;

/*!
 * \class LIFTimeDrivenModel_1_4_NEW
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with a membrane potential and
 * four conductances.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date May 2013
 */
class LIFTimeDrivenModel_1_4_NEW : public TimeDrivenNeuronModel {
	protected:

		/*!
		 * \brief table values
		 * gnmdainf
		 */
		static float * channel_values;
		static const float Max_V;
		static const float Min_V;
		static const int TableSize=1024*1024;
		static const float aux;

		/*!
		 * \brief Excitatory reversal potential in mV units
		 */
		float eexc;

		/*!
		 * \brief Inhibitory reversal potential in mV units
		 */
		float einh;

		/*!
		 * \brief Resting potential in mV units
		 */
		float erest;

		/*!
		 * \brief Firing threshold in mV units
		 */
		float vthr;

		/*!
		 * \brief Membrane capacitance in pF units
		 */
		float cm;
		float inv_cm;

		/*!
		 * \brief AMPA receptor time constant in ms units
		 */
		float tampa;
		float inv_tampa;

		
		/*!
		 * \brief GABA receptor time constant in ms units
		 */
		float tinh;
		float inv_tinh;

		/*!
		 * \brief NMDA receptor time constant in ms units
		 */
		float tnmda;
		float inv_tnmda;

		/*!
		 * \brief Gap Junction time constant in ms units
		 */
		float tgj;
		float inv_tgj;

		/*!
		 * \brief Refractory period in ms units
		 */
		float tref;

		/*!
		 * \brief Resting conductance in nS units
		 */
		float grest;

		/*!
		 * \brief Gap junction factor in mV/nS units
		 */
		float fgj;


		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void LoadNeuronModel(string ConfigFile) ;


		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index The cell index inside the VectorNeuronState.
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		void SynapsisEffect(int index, Interconnection * InputConnection);

	public:

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=5;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=1;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		static const int N_TimeDependentNeuronState=4;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model identificator.
		 * \param NeuronModelID Neuron model configuration file.
		 */
		LIFTimeDrivenModel_1_4_NEW(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~LIFTimeDrivenModel_1_4_NEW();


		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
 		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() ;


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
		 * \param target the neuron which receives the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, Neuron * target, double time);


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
		virtual bool EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale);


		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 * \param index Neuron index inside the VectorNeuronState
		 */
		virtual void EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time);


		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 * \param elapsed_time_index index inside the conductance_exp_values array.
		 */
		virtual void EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time, int elapsed_time_index);



		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseTypeNumber(int Type);


		static float * Generate_channel_values(){
			float * NewLookUpTable=new float[TableSize];
			for(int i=0; i<TableSize; i++){
				float V = Min_V + ((Max_V-Min_V)*i)/(TableSize-1);
				
				//gnmdainf
				float gnmdainf = 1.0f/(1.0f + exp(-0.062f*V)*(1.2f/3.57f));
				NewLookUpTable[i]=gnmdainf;
			}
			return NewLookUpTable;
		}


		static float * Get_channel_values(float value){
				int position=int((value-Min_V)*aux);
				if(position<0){
					position=0;
				}else if(position>(TableSize-1)){
					position=TableSize-1;
				}
				return (channel_values + position);
		} 


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
		void CalculateElectricalCouplingSynapseNumber(Interconnection * inter){};

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		void InitializeElectricalCouplingSynapseDependencies(){};

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter){};

};

#endif /* LIFTIMEDRIVENMODEL_1_4_NEW_H_ */
