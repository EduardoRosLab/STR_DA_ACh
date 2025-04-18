/***************************************************************************
 *                           LIFTimeDrivenModel_1_2_GPU_NEW.h              *
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

#ifndef LIFTIMEDRIVENMODEL_1_2_GPU_NEW_H_
#define LIFTIMEDRIVENMODEL_1_2_GPU_NEW_H_

/*!
 * \file LIFTimeDrivenModel_1_2_GPU_NEW.h
 *
 * \author Francisco Naveros
 * \date October 2016
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model with one 
 * differential equation and two time dependent equations (conductances). This model is
 * implemented in CPU to control a GPU class.This class is a redefinition of 
 * LIFTimeDrivenModel_1_2_GPU neuron model using mV, ms, nS and pF units.
 */

#include "./TimeDrivenNeuronModel_GPU.h"

#include <string>



using namespace std;

class InputSpike;
class VectorNeuronState;
class VectorNeuronState_GPU;
class Interconnection;

class LIFTimeDrivenModel_1_2_GPU2_NEW;


/*!
 * \class LIFTimeDrivenModel_1_2_GPU_NEW
 *
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model with a membrane potential and
 * two conductances. This model is implemented in CPU to control a GPU class.
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date October 2016
 */
class LIFTimeDrivenModel_1_2_GPU_NEW : public TimeDrivenNeuronModel_GPU {
	protected:
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

		/*!
		 * \brief AMPA receptor time constant in ms units
		 */
		float texc;

		/*!
		 * \brief GABA receptor time constant in ms units
		 */
		float tinh;

		/*!
		 * \brief Refractory period in ms units
		 */
		float tref;

		/*!
		 * \brief Resting conductance in nS units
		 */
		float grest;


		/*!
		 * \brief Neuron model in the GPU.
		*/
		LIFTimeDrivenModel_1_2_GPU2_NEW ** NeuronModel_GPU2;


		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel(string ConfigFile) ;


		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index The cell index inside the VectorNeuronState_GPU.
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index,VectorNeuronState_GPU * state, Interconnection * InputConnection);

	public:


		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=3;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=1;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		static const int N_TimeDependentNeuronState=2;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model identificator.
		 * \param NeuronModelID Neuron model configuration file.
		 */
		LIFTimeDrivenModel_1_2_GPU_NEW(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~LIFTimeDrivenModel_1_2_GPU_NEW();


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
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, VectorNeuronState * State, double CurrentTime);


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
		virtual bool CheckSynapseTypeNumber(int Type);

};

#endif /* LIFTIMEDRIVENMODEL_1_2_GPU_NEW_H_ */
