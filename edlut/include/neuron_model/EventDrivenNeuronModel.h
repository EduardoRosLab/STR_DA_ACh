/***************************************************************************
 *                           EventDrivenNeuronModel.h                      *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef EVENTDRIVENNEURONMODEL_H_
#define EVENTDRIVENNEURONMODEL_H_

/*!
 * \file EventDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which abstracts an event-driven neuron model.
 */

#include "./NeuronModel.h"

#include "simulation/NetworkDescription.h"
#include <cstring>

using namespace std;

class Neuron;
class EndRefractoryPeriodEvent;


/*!
 * \class EventDrivenNeuronModel
 *
 * \brief Event-Driven Spiking neuron model
 *
 * This class abstracts the behavior of an event-driven neuron model in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class EventDrivenNeuronModel : public NeuronModel {
	public:


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		EventDrivenNeuronModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~EventDrivenNeuronModel();

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		virtual VectorNeuronState * InitializeState() = 0;


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
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time) = 0;

		/*!
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*/
		virtual void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
		}

		/*!
		 * \brief It predicts if the neuron would generate a internalSpike aftern all the propagated spikes have arrived. 
		 *
		 * It predicts if the neuron would generate a internalSpike after all the propagated spikes have arrived. 
		 *
		 * \param target Neuron that must be updated.
		 * \param time time.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessActivityAndPredictSpike(Neuron * target, double time) = 0;


		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron *  Cell) = 0;

		/*!
		 * \brief It processes an internal spike (generated spike in the cell).
		 *
		 * It processes an internal spike (generated spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated (output) spike. It must be externally done.
		 * \note Before generating next spike, you should check if this spike must be discard.
		 *
		 * \see DiscardSpike
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual EndRefractoryPeriodEvent * ProcessInternalSpike(InternalSpike * OutputSpike)=0;

		virtual InternalSpike * GenerateNextSpike(double time, Neuron * neuron)=0;

		/*!
		 * \brief Check if the spike must be discard.
		 *
		 * Check if the spike must be discard. A spike must be discard if there are discrepancies between
		 * the next predicted spike and the spike time.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return True if the spike must be discard. False in otherwise.
		 */
		virtual bool DiscardSpike(InternalSpike *  OutputSpike) = 0;

		/*!
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		enum NeuronModelSimulationMethod GetModelSimulationMethod();

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		enum NeuronModelType GetModelType();

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		virtual enum NeuronModelOutputActivityType GetModelOutputActivityType() = 0;
		
		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		virtual enum NeuronModelInputActivityType GetModelInputActivityType() = 0;

		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection) = 0;

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

		/*!
		* \brief It initialieses the structure required to implement external input current synapsis.
		*
		* It initialieses the structure required to implement external input current synapsis.
		*/
		virtual void InitializeInputCurrentSynapseStructure(){};

		/*!
		 * \brief It returns the neuron model parameters.
		 *
		 * It returns the neuron model parameters.
		 *
		 * \returns A dictionary with the neuron model parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters() const;

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
		virtual std::map<std::string, boost::any> GetSpecificNeuronParameters(int index) const  = 0;

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
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();


        /*!
         * \brief Comparison operator between neuron models.
         *
         * It compares two neuron models.
         *
         * \return True if the neuron models are of the same type and with the same parameters.
         */
		virtual bool compare(const NeuronModel * rhs) const{
			if (!NeuronModel::compare(rhs)){
				return false;
			}
			const EventDrivenNeuronModel * e = dynamic_cast<const EventDrivenNeuronModel *> (rhs);
			if (e == 0) return false;

			return true;
		};
};

#endif /* EVENTDRIVENNEURONMODEL_H_ */
