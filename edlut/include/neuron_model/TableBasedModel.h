/***************************************************************************
 *                           TableBasedModel.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef TABLEBASEDMODEL_H_
#define TABLEBASEDMODEL_H_

/*!
 * \file TableBasedModel.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which implements a neuron model based in
 * look-up tables.
 */

#include "EventDrivenNeuronModel.h"

#include "../spike/EDLUTFileException.h"

class NeuronModelTable;
class Interconnection;

/*!
* \brief Maximum number of state variables that the TableBasedModel can manage inside a neuron model.
*/
#define MAX_VARIABLES 10

/*!
 * \class TableBasedModel
 *
 * \brief Spiking neuron model based in look-up tables
 *
 * This class implements the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This behavior is calculated based in precalculated look-up tables.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class TableBasedModel: public EventDrivenNeuronModel {
	protected:
		/*!
		 * \brief Number of state variables (no include time).
		 */
		unsigned int NumStateVar;

		/*!
		 * \brief Number of time dependent state variables.
		 */
		unsigned int NumTimeDependentStateVar;

		/*!
		 * \brief Number of synaptic variables.
		 */
		unsigned int NumSynapticVar;

		/*!
		 * \brief Index of synaptic variables.
		 */
		unsigned int * SynapticVar;

		/*!
		 * \brief Order of state variables.
		 */
		unsigned int * StateVarOrder;

		/*!
		 * \brief Table which calculates each state variable.
		 */
		NeuronModelTable ** StateVarTable;

		/*!
		 * \brief Firing time table
		 */
		NeuronModelTable * FiringTable;

		/*!
		 * \brief End firing time table
		 */
		NeuronModelTable * EndFiringTable;

		/*!
		 * \brief Number of tables
		 */
		unsigned int NumTables;

		/*!
		 * \brief Precalculated tables
		 */
		NeuronModelTable * Tables;


		/*!
		 * \brief Vector where we temporary store initial values
		 */
		float * InitValues;

		/*!
		* \brief String where is stored the name of the configuration file where are stored the look-up table parameters ("file.cfg").
		*/
		string conf_filename;

		/*!
		* \brief String where is stored the name of the file where are stored the look-up tables ("file.dat").
		*/
		string tab_filename;

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
		 * \brief It loads the neuron model tables.
		 *
		 * It loads the neuron model tables from his .dat associated file.
		 *
		 * \pre The neuron model must be previously initialized or loaded
		 *
		 * \param TableFile Name of the table file (*.dat).
		 *
		 * \see LoadNeuronModel()
		 * \throw EDLUTException If something wrong has happened in the tables loads.
		 */
		virtual void LoadTables(string TableFile) ;

		/*!
		 * \brief It returns the end of the refractory period.
		 *
		 * It returns the end of the refractory period.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 *
		 * \return The end of the refractory period. -1 if no spike is predicted.
		 */
		virtual double EndRefractoryPeriod(int index, VectorNeuronState * State);

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 * \param CurrentTime Current simulation time.
		 */
		virtual void UpdateState(int index, VectorNeuronState * State, double CurrentTime);

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index, Interconnection * InputConnection);


		/*!
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param index index inside the VectorNeuronState of the neuron model.
		 * \param VectorNeuronState of the neuron model.
		 *
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(int index, VectorNeuronState * State);

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new default neuron model object. The configuration parameters and look-up table will be loaded in other function.
		 *
		 */
		TableBasedModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~TableBasedModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() ;

		/*!
		 * \brief It creates the neuron state and initializes to defined values.
		 *
		 * It creates the neuron state and initializes to defined values.
		 *
		 * \return A new object with the neuron state.
		 */
		virtual VectorNeuronState * InitializeState();

		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron *  Cell);


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
		 * \brief It predicts if the neuron would generate a internalSpike after the ToleranceTime. 
		 *
		 * It predicts if the neuron would generate a internalSpike after the ToleranceTime. 
		 *
		 * \param target Neuron that must be updated.
		 * \param time time.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessActivityAndPredictSpike(Neuron * target, double time);


		/*!
		 * \brief It processes an internal spike and generates an end refractory period event.
		 *
		 * It processes an internal spike and generates an end refractory period event.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new end refractory period event.
		 */
		virtual EndRefractoryPeriodEvent * ProcessInternalSpike(InternalSpike * OutputSpike);

		/*!
		 * \brief It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * It calculates if an internal spike must be generated at the end of the refractory period.
		 *
		 * \param time end of the refractory period.
		 * \param neuron source neuron.
		 *
		 * \return A new internal spike.
		 */
		virtual InternalSpike * GenerateNextSpike(double time, Neuron * neuron);

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
		virtual bool DiscardSpike(InternalSpike *  OutputSpike);


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
		 * \brief It prints the table based model info.
		 *
		 * It prints the current table based model characteristics.
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
		 * \brief It returns the default parameters of the neuron model.
		 *
		 * It returns the default parameters of the neuron models. It may be used to obtained the parameters that can be
		 * set for this neuron model.
		 *
		 * \returns A dictionary with the neuron model default parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();

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
            if (!EventDrivenNeuronModel::compare(rhs)){
                return false;
            }
            const TableBasedModel * e = dynamic_cast<const TableBasedModel *> (rhs);
            if (e == 0) return false;

			return this->NumStateVar == e->NumStateVar &&
				this->NumTimeDependentStateVar == e->NumTimeDependentStateVar &&
				this->NumSynapticVar == e->NumSynapticVar &&
				this->SynapticVar == e->SynapticVar &&
				this->StateVarOrder == e->StateVarOrder &&
				this->StateVarTable == e->StateVarTable &&
				this->FiringTable == e->FiringTable &&
				this->EndFiringTable == e->EndFiringTable &&
				this->NumTables == e->NumTables &&
				this->Tables == e->Tables &&
				this->InitValues == e->InitValues;
        };
};

#endif /* TABLEBASEDMODEL_H_ */
