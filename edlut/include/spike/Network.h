/***************************************************************************
 *                           Network.h                                     *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NETWORK_H_
#define NETWORK_H_

/*!
 * \file Network.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \author Francisco Naveros
 * \date August 2008
 *
 * \note Modified on January 2011 in order to include time-driven simulation support.
 * New state variables (ntimedrivenneurons and timedrivenneurons)
 *
 * \note Modified on January 2012 in order to include time-driven simulation support in GPU.
 * New state variables (timedrivenneurons, ntimedrivenneurons, ntimedrivenneurons_GPU and
 * timedrivenneurons_GPU)
 *
 * \note Modified on
 *
 * This file declares a class which abstracts a spiking neural network.
 */

#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <list>

#include "./EDLUTFileException.h"

#include "../simulation/PrintableObject.h"

class Interconnection;
class NeuronModel;
class Neuron;
class LearningRule;
class EventQueue;
struct NeuronLayerDescription;
struct ModelDescription;
struct SynapticLayerDescription;

using namespace std;
/*!
 * \class Network
 *
 * \brief Spiking neural network
 *
 * This class abstract the behaviour of a spiking neural network.
 * It is composed by interconnections, neurons, neuron types and weight changes.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \author Francisco Naveros
 * \date August 2008
 */
class Network : public PrintableObject{
	private:

		/*!
		 * \brief Network interconnections.
		 */
		Interconnection *inters;

   		/*!
   		 * \brief Number of interconnections.
   		 */
   		long int ninters;

   		/*!
   		 * \brief Neuron types.
   		 */
   		NeuronModel *** neutypes;

   		/*!
   		 * \brief Neuron types number.
   		 */
   		int nneutypes;

   		/*!
   		 * \brief Neuron array.
   		 */
   		Neuron *neurons;

   		/*!
   		 * \brief Number of neurons.
   		 */
   		int nneurons;

		/*!
		 * \brief Time-driven cell (model) array.
		 */
		Neuron **** timedrivenneurons;

		/*!
   		 * \brief Number of time-driven neurons.
   		 */
		int ** ntimedrivenneurons;

		/*!
		 * \brief Time-driven cell (model) arrays in GPU.
		 */
		Neuron **** timedrivenneurons_GPU;

		/*!
		 * \brief Number of time-driven neurons for every model in GPU.
		 */
		int ** ntimedrivenneurons_GPU;


   		/*!
   		 * \brief Learning rules.
   		 */
   		LearningRule ** wchanges;

   		/*!
   		 * \brief Number of learning rules.
   		 */
   		unsigned int nwchanges;

   		/*!
   		 * \brief Initial connection ordenation.
   		 */
   		Interconnection ** wordination;

		/*!
 		 * Number of OpenMP thread.
 		 */
		int NumberOfQueues;

		/*!
 		 * Minimun propagation delay. Each propagation delay will be rounded to a multiple of this value.
 		 */
		double minpropagationdelay;

		/*!
 		 * Inverse of minpropagationdelay.
 		 */
		double invminpropagationdelay;

   		/*!
   		 * \brief It sorts the connections by the source neuron and the delay and add the output connections
   		 *
   		 * It sorts the connections by the source neuron (from the lowest to the highest index) and by the connection
   		 * delay. It adds the connections to the output connections of the source neuron.
   		 *
   		 * \post The connections will be sorted by source neuron and delay.
   		 */
   		void FindOutConnections();

		//void FindOutConnections(int N_LearningRule, int * typeLearningRule);

   		/*!
   		 * \brief It adds the input connection to the target neuron.
   		 *
   		 * It adds the connections to the input connections of the target neuron.
   		 */
   		void FindInConnections();

		//void FindInConnections(int N_LearningRule, int * typeLearningRule);

   		/*!
   		 * \brief It sorts the connections by the connection index.
   		 *
   		 * It sorts the connections by the connection index. This ordination
   		 * is the ordenation needed for the weight load. The ordination will be
   		 * in wordination field.
   		 */
   		void SetWeightOrdination();

   		/*!
  		 * \brief It prints information about load tables.
  		 *
  		 * It prints information about load tables.
  		 *
  		 */
  		void TablesInfo();

  		/*!
  		 * \brief It prints information about load types.
  		 *
  		 * It prints information about load types.
  		 *
  		 */
  		void TypesInfo();

   		/*!
   		 * \brief It loads the neuron type characteristics.
   		 *
   		 * It checks if the neuron type has been loaded, and in other case,
   		 * it loads the characteristics from the neuron type files.
   		 *
   		 * \param ident_type Type of the neuron model. At this moment, only "SRMTimeDriven" and "TableBasedModel" are implemented.
   		 * \param neutype The name of the neuron type to load.
		 * \param ni Index of the neuron type
		 * \param Currentline current line inside the network file
		 * \param netfile network file name
   		 *
   		 * \return The loaded (or existing) neuron type.
   		 * \throw EDLUTException If the neuron model file hasn't been able to be correctly readed.
   		 */
		NeuronModel ** LoadNetTypes(string ident_type, string neutype, int & ni) ;

  		/*!
  		 * \brief It Initialize all Vector Neuron State.
  		 *
  		 * It Initialize all Vector Neuron State.
  		 *
		 * \param N_neurons Neuron number for each neuron model.
  		 */
		void InitializeStates(int ** N_neurons);

   		/*!
   		 * \brief It inits the spikes predictions of every neuron in the network.
   		 *
   		 * It adds all the spike predictions in the network in the initial conditions.
   		 *
   		 * \param Queue The event queue where the spikes will be added.
   		 */
   		void InitNetPredictions(EventQueue * Queue);

		/*!
		 * \brief It creates a network from the parameters.
		 *
		 * It loads a new network from the parameter description.
		 *
		 * \param neuron_layer_list The list of neuron layer descriptions.
		 * \param learning_rule_list The list of learning rule descriptions.
		 * \param synaptic_layer_list The list of synaptic layer descriptions.
		 *
		 */
		void LoadNet(const std::list<NeuronLayerDescription> & neuron_layer_list,
					 const std::list<ModelDescription> & learning_rule_list,
					 const std::list<SynapticLayerDescription> & synaptic_layer_list) ;

   		/*!
   		 * \brief It initializes the weight change structures from the representation.
   		 *
   		 * It initializes the weight change structures according to the parameter
   		 *
   		 * \param learning_rule_list List of learning rules to initialize
   		 *
   		 * \return Vector with the number of synapses associated to each learning rule
   		 */
        std::vector<int> CreateWeightChanges(const list<ModelDescription> &learning_rule_list);

		/*!
		* \brief It initializes the neurons of the network.
		*
		* It initializes the neuronal structures according to the parameter
		*
		* @param neuron_layer_list List of neuron layers to initialize
		*/
		void CreateNeuronLayers(const list<NeuronLayerDescription> &neuron_layer_list);

		/*!
		* \brief It initializes the synapses of the model.
		*
		* It initializes the synapses according to the parameter
		*
		* \param synaptic_layer_list List of synaptic layers to initialize
		* \param Vector with the number of synapses associated to each learning rule
		*/
	 	void CreateConnections(const list<SynapticLayerDescription> &synaptic_layer_list, std::vector<int> & NConnectionsPerLearning);

		/*!
		* \brief It initializes the learning rule states.
		*
		* It initializes the synapses according to the parameter
		*
		* \param synaptic_layer_list List of synaptic layers to initialize
		*/
		void InitializeLearningRuleState(const vector<int> &N_ConectionWithLearning) const;

   	public:


		/*!
		* This variable sets if at least one neuron is monitored
		*/
		bool monitore_neurons;



		/*!
		 * \brief It loads the network configuration from a file.
		 *
		 * It loads a new network from files and creates the description objects.
		 *
		 * \param netfile The file name of the network configuration file.
		 * \param wfile The file name of the weight configuration file.
		 * \param neuron_layer_list The returned neuron layer description list.
		 * \param learning_rule_list The returned learning rule description list.
		 * \param synaptic_layer_list The returned synaptic layer description list.
		 *
		 * \note The returning objects can then be used with the LoadNet to effectively creates the network
		 * before simulation.
		 *
		 * \throw EDLUTFileException If the network configuration file hasn't been able to be correctly readed.
		 */
		static void ParseNet(const char *netfile, const char *wfile,
							 std::list<NeuronLayerDescription> & neuron_layer_list,
							 std::list<ModelDescription> & learning_rule_list,
							 std::list<SynapticLayerDescription> & synaptic_layer_list) ;

		/*!
        * \brief It creates a new network object by from the intermediate network representation objects.
        *
        * It creates a new network object. The network is created from the network representation objects.
        * Finally, it initializes the event queue with the initial spikes.
        *
        * \param neuron_layer_list The list of neuron layers.
        * \param learning_rule_list The list of learning rules.
        * \param synaptic_layer_list The list of synaptic layers.
        * \param Queue The event queues where the events will be inserted.
        * \param numberOfQueues The number of simulation queues.
        *
        * \throw EDLUTException If some error has happened.
        */
		Network(const std::list<NeuronLayerDescription> & neuron_layer_list,
				const std::list<ModelDescription> & learning_rule_list,
				const std::list<SynapticLayerDescription> & synaptic_layer_list,
				EventQueue * Queue, int numberOfQueues) ;
   		/*!
   		 * \brief Default destructor.
   		 *
   		 * It destroies a network object and frees the memory.
   		 */
   		~Network();

   		/*!
   		 * \brief It gets a neuron by the index.
   		 *
   		 * It returns a neuron from the index.
   		 *
   		 * \param index The index of the neuron to get.
   		 *
   		 * \return The neuron whose index is the parameter.
   		 */
   		Neuron * GetNeuronAt(int index) const;

   		/*!
   		 * \brief It gets the number of neurons in the network.
   		 *
   		 * It gets the number of neurons in the network.
   		 *
   		 * \return The number of neurons.
   		 */
   		int GetNeuronNumber() const;

		/*!
   		 * \brief It gets a time-driven neuron by the index0 and index1.
   		 *
   		 * It returns a time-driven neuron from array index0 and position index1.
   		 *
   		 * \param index0 The array of the time-driven neuron to use.
		 * \param index1 The index of the time-driven neuron to get.
		 *
   		 * \return The time-driven neuron whose index is the parameter index1 in array index0.
   		 */
		Neuron ** GetTimeDrivenNeuronAt(int index0, int index1) const;

		/*!
   		 * \brief It gets a time-driven neuron by the index0 and index1.
   		 *
   		 * It returns a time-driven neuron from array index0 and position index1.
   		 *
   		 * \param index0 The array of the time-driven neuron to use.
		 * \param index1 The index of the time-driven neuron to get.
		 *
   		 * \return The time-driven neuron whose index is the parameter index1 in array index0.
   		 */
		Neuron * GetTimeDrivenNeuronAt(int index0, int index1, int index2) const;

		/*!
   		 * \brief It gets a time-driven neuron in GPU by the index0 and index1.
   		 *
   		 * It returns a time-driven neuron in GPU from array index0 and position index1.
   		 *
		 * \param index0 The array of the OpenMP queue.
   		 * \param index1 The array of the time-driven neuron to use.
		 *
   		 * \return The time-driven neuron vector
   		 */
		Neuron ** GetTimeDrivenNeuronGPUAt(int index0, int index1) const;

		/*!
   		 * \brief It gets a time-driven neuron in GPU by the index0 and index1.
   		 *
   		 * It returns a time-driven neuron in GPU from array index0 and position index1.
   		 *
		 * \param index0 The array of the OpenMP queue.
   		 * \param index1 The array of the time-driven neuron to use.
		 * \param index2 The index of the time-driven neuron to get.
		 *
   		 * \return The time-driven neuron
   		 */
		Neuron * GetTimeDrivenNeuronGPUAt(int index0, int index1, int index2) const;

		/*!
		 * \brief It gets the numbers of time-driven neurons for every model in the network.
		 *
		 * It gets the numbers of time-driven neurons for every model in the network
		 *
		 * \return the numbers of time-driven neurons for every model in the network
		 */
		int ** GetTimeDrivenNeuronNumber() const;

		/*!
		 * \brief It gets the numbers of time-driven neurons in GPU for every model in the network.
		 *
		 * It gets the numbers of time-driven neurons in GPU for every model in the network
		 *
		 * \return the numbers of time-driven neurons in GPU for every model in the network
		 */
		int ** GetTimeDrivenNeuronNumberGPU() const;

		/*!
		 * \brief It gets the number of neuron model in the network.
		 *
		 * It gets the number of neuron model in the network.
		 *
		 * \return The number of neuron model.
		 */
		int GetNneutypes() const;

		 /*!
		 * \brief It gets a neuron model by the index.
		 *
		 * It returns a neuron model from the index.
		 *
		 * \param index The index of the neuron model to get.
		 *
		 * \return The neuron model whose index is the parameter.
		 */
		NeuronModel ** GetNeuronModelAt(int index) const;


		/*!
		 * \brief It gets a neuron model by the index.
		 *
		 * It returns a neuron model from the index.
		 *
		 * \param index The index of the neuron model to get.
		 *
		 * \return The neuron model whose index is the parameter.
		 */
		NeuronModel * GetNeuronModelAt(int index1, int index2) const;


   		/*!
		 * \brief It gets a learning rule by the index.
		 *
		 * It returns a learning rule from the index.
		 *
		 * \param index The index of the learning rule to get.
		 *
		 * \return The rule whose index is the parameter.
		 */
		LearningRule * GetLearningRuleAt(int index) const;

		/*!
		 * \brief It gets the number of learning rules in the network.
		 *
		 * It gets the number of learning rules in the network.
		 *
		 * \return The number of learning rules.
		 */
		int GetLearningRuleNumber() const;

   		/*!
   		 * \brief It saves the weights in a file.
   		 *
   		 * It saves the network weights in a file.
   		 *
   		 * \param wfile The file name where we save the weights.
   		 *
   		 * \throw EDLUTException If some error happends.
   		 */
		void SaveWeights(const char *wfile) ;

		/*!
		* \brief It returns all the synaptic weigths in a compressed format with two vector.
		*
		* It returns all the synaptic weigths in a compressed format with two vector.
		*
		* \param N_equal_weights vector storing in each value the number of synapses with the same weight.
		* \param equal_weights vector storign in each value the weight of the synapses with the same weights.
		*/
		void GetCompressedWeights(std::vector<int> & N_equal_weights, std::vector<float> & equal_weights);

		/*!
		* \brief It returns all the synaptic weigths in a extended format with a vector.
		*
		* It returns all the synaptic weigths in a extended format with a vector.
		*
		* \return vector with the synaptic weights.
		*/
		std::vector<float> GetWeights();

		/*!
		* \brief It returns the solicited synaptic weigths in a extended format with a vector.
		*
		* It returns the solicited synaptic weigths in a extended format with a vector.
		*
		* \param synaptic_indexes synaptic indexes
		*
		* \return vector with the solicityed synaptic weights.
		*/
		std::vector<float> GetSelectedWeights(std::vector<int> synaptic_indexes);

   		/*!
   		 * \brief It prints the network info.
   		 *
   		 * It prints the current network characteristics (number of neurons, types, connections...).
   		 *
   		 * \param out The stream where it prints the information.
   		 *
   		 * \return The stream after the printer.
   		 */
   		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It gets the number of OpenMP thread.
		 *
		 * It gets the number of OpenMP thread.
		 *
		 * \return The number of OpenMP thread.
		 */
		int GetNumberOfQueues();

		/*!
		 * \brief It calculate the minimun propagation delay between neurons of different OpenMP queues. This time is the maximun value that can be used in the synchronization between OpenMP queues.
		 *
		 * It calculate the minimun propagation delay between neurons of different OpenMP queues.
		 *
		 * \return The minimun propagation delay between neurons of different OpenMP queues.
		 */
		double GetMinInterpropagationTime();


		/*!
		 * \brief It rounds the propagation delay to a value multiple of "minpropagationdelay".
		 *
		 * It rounds the propagation delay to a value multiple of "minpropagationdelay".
		 *
		  * \param time propagation delay.
		 *
		 * \return A propagation delay multiple of "minpropagationdelay".
		 */
		double RoundPropagationDelay(double time);

		/*!
		* \brief It calculates the electrical coupling dependencies for neuron model such as the "TimeDrivenInferiorOliveCell".
		*
		* It calculates the electrical coupling dependencies for neuron model such as the "TimeDrivenInferiorOliveCell".
		*/
		void CalculaElectricalCouplingDepedencies();

};

/*!
 * \brief It sorts two connections by the source neuron and the delay.
 *
 * This functions sorts two connections by the source neuron and the delay.
 *
 * \param e1 The firs connection.
 * \param e2 The second connection.
 *
 * \return 0 if the two connections have the same index of the source neuron and the same delay. <0 if
 * the second connection have a higher index of the source neuron or the same index and lower delay.
 * >0 if the first connection have a higher index of the source neuron or the same index and lower delay.
 */
int qsort_inters(const void *e1, const void *e2);


#endif /*NETWORK_H_*/
