/***************************************************************************
 *                           ESTDEWeightChange.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2022 by Álvaro González-Redondo                        *
 * email                : alvarogr@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef ESTDEWEIGHTCHANGE_H_
#define ESTDEWEIGHTCHANGE_H_

#include "./WithPostSynaptic.h"

#include "../simulation/NetworkDescription.h"

/*!
 * \file ESTDEWeightChange.h
 *
 * \author Álvaro González-Redondo
 * \date December 2022
 *
 * This file declares a class which abstracts a ESTDE learning rule.
 */

class Interconnection;

/*!
 * \class ESTDEWeightChange
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a ESTDE learning rule.
 *
 * \author Álvaro González-Redondo
 * \date December 2022
 */
class ESTDEWeightChange: public WithPostSynaptic {
	public:
		static const int N_TRIGGERS = 4;

	protected:
		/*!
		* \brief Maximum weight change for LTP
		*/
    	float pre_dif;
		float prepos_dif;
		float prepos_tri_dif[N_TRIGGERS];

		/*!
		* \brief Decay parameter for LTP
		*/
		float tau_pre;

		/*!
		* \brief Maximum weight change LTD
		*/
    	float pos_dif;
		float pospre_dif;
		float pospre_tri_dif[N_TRIGGERS];

		/*!
		 * \brief Decay parameter LTD
		 */
		float tau_pos;

		/*!
		 * \brief Decay parameter eligibility
		 */
		float tau_eli;

		/*!
		 * \brief Decay parameter eligibility
		 */
		float tau_ach;
		float ach_tri_dif[N_TRIGGERS];

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule.
		 */
		ESTDEWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~ESTDEWeightChange();


		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		virtual void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons);


		/*!
		 * \brief It gets the maximum value of the weight change for LTD.
		 *
		 * It gets the maximum value of the weight change for LTD.
		 *
		 * \return The maximum value of the weight change for LTD.
		 */
		// float GetMaxWeightChangeLTD() const;

		/*!
		 * \brief It sets the maximum value of the weight change for LTD.
		 *
		 * It sets the maximum value of the weight change for LTD.
		 *
		 * \param NewMaxChange The new maximum value of the weight change for LTD.
		 */
		// void SetMaxWeightChangeLTD(float NewMaxChange);

		/*!
		 * \brief It gets the maximum value of the weight change for LTP.
		 *
		 * It gets the maximum value of the weight change for LTP.
		 *
		 * \return The maximum value of the weight change for LTP.
		 */
		// float GetMaxWeightChangeLTP() const;

		/*!
		 * \brief It sets the maximum value of the weight change for LTP.
		 *
		 * It sets the maximum value of the weight change for LTP.
		 *
		 * \param NewMaxChange The new maximum value of the weight change for LTP.
		 */
		// void SetMaxWeightChangeLTP(float NewMaxChange);


		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 *
		 * \return The learning rule description object.
		 *
		 * \throw EDLUTException If something wrong happens in reading the learning rule properties.
		 */
		static ModelDescription ParseLearningRule(FILE * fh) ;

		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);

		/*!
		* \brief It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* \param neuron The target neuron that manage the postsynaptic spike
		* \param SpikeTime The spike time of the postsynaptic spike.
		*/
		virtual void ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime);

		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It creates a new learning rule object of this type.
		 *
		 * It creates a new learning rule object of this type.
		 *
		 * \param param_map The learning rule description object.
		 *
		 * \return A newly created ExpWeightChange object.
		 */
		static LearningRule* CreateLearningRule(ModelDescription lrDescription);

		/*!
		 * \brief It provides the name of the learning rule
		 *
		 * It provides the name of the learning rule, i.e. the name that can be mentioned to use this learning rule.
		 *
		 * \return The name of the learning rule
		 */
		static std::string GetName(){
			return "ESTDE";
		};

		/*!
		 * \brief It returns the learning rule parameters.
		 *
		 * It returns the learning rule parameters.
		 *
		 * \returns A dictionary with the learning rule parameters
		 */
		virtual std::map<std::string,boost::any> GetParameters();

		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties from parameter map.
		 *
		 * \param param_map The dictionary with the learning rule parameters.
		 *
		 * \throw EDLUTFileException If it happens a mistake with the parameters in the dictionary.
		 */
		virtual void SetParameters(std::map<std::string, boost::any> param_map) ;

		/*!
		 * \brief It returns the default parameters of the learning rule.
		 *
		 * It returns the default parameters of the learning rule. It may be used to obtained the parameters that can be
		 * set for this learning rule.
		 *
		 * \returns A dictionary with the learning rule parameters.
		 */
		static std::map<std::string,boost::any> GetDefaultParameters();


};

#endif /* ESTDEWEIGHTCHANGE_H_ */
