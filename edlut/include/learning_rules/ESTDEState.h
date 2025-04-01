/***************************************************************************
 *                           ESTDEState.h                                   *
 *                           -------------------                           *
 * copyright            : (C) 2022 by Álvaro González-Redondo                     *
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

#ifndef ESTDESTATE_H_
#define ESTDESTATE_H_

#include "ConnectionState.h"

/*!
 * \file ESTDEState.h
 *
 * \author Álvaro González-Redondo
 * \date October 2022
 *
 * This file declares a class which abstracts the current state of a synaptic connection
 * with ESTDE capabilities.
 */

/*!
 * \class ESTDEState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection including ESTDE and defines the state variables of
 * that connection.
 *
 * \author Álvaro González-Redondo
 * \date October 2022
 */

class ESTDEState : public ConnectionState{

	public:
		/*!
		 * LTP time constant.
		 */
		float tau_pre, inv_tau_pre;

		/*!
		 * LTD time constant.
		 */
		float tau_pos, inv_tau_pos;

		/*!
		 * Eligibility traces parameters.
		 */
		float tau_eli, inv_tau_eli;

		/*!
		 * ACh trace parameters
		 */
		float tau_ach, inv_tau_ach;

		//index states
		static const int N_STATES = 5;
		static const int PRE_INDEX = 0;
		static const int POS_INDEX = 1;
		static const int PREPOS_INDEX = 2;
		static const int POSPRE_INDEX = 3;
		static const int ACH_INDEX = 4;
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param LTPtau Time constant of the LTP component.
		 * \param LTDtau Time constant of the LTD component.
		 */
		ESTDEState(
			int n_syn,
			float new_tau_pre,
			float new_tau_pos,
			float new_tau_eli,
			float new_tau_ach
		);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~ESTDEState();

		/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \return The accumulated presynaptic activity.
		 */
		//float GetPresynapticActivity(unsigned int index);
		inline float GetPresynapticActivity(unsigned int index){
			return this->GetStateVariableAt(index, PRE_INDEX);
		}

		/*!
		 * \brief It gets the value of the accumulated postsynaptic activity.
		 *
		 * It gets the value of the accumulated postsynaptic activity.
		 *
		 * \return The accumulated postsynaptic activity.
		 */
		//float GetPostsynapticActivity(unsigned int index);
		inline float GetPostsynapticActivity(unsigned int index){
			return this->GetStateVariableAt(index, POS_INDEX);
		}

		inline float GetEligibilityPrePos(unsigned int index){
			return this->GetStateVariableAt(index, PREPOS_INDEX);
		}

		inline float GetEligibilityPosPre(unsigned int index){
			return this->GetStateVariableAt(index, POSPRE_INDEX);
		}

		inline float GetACh(unsigned int index){
			return this->GetStateVariableAt(index, ACH_INDEX);
		}

		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position Position inside each connection.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int index, unsigned int position);

		/*!
		 * \brief set new time to spikes.
		 *
		 * It set new time to spikes.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param NewTime new time.
		 * \param pre_post In some learning rules (i.e. STDPLS) this variable indicate wether the update affects the pre- or post- variables.
		 */
		virtual void SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post);


		/*!
		 * \brief It implements the behaviour when it transmits a spike.
		 *
		 * It implements the behaviour when it transmits a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPresynapticSpike(unsigned int index);

		/*!
		 * \brief It implements the behaviour when the target cell fires a spike.
		 *
		 * It implements the behaviour when it the target cell fires a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPostsynapticSpike(unsigned int index);



		virtual void SetWeight(unsigned int index, float weight, float max_weight);

};

#endif /* NEURONSTATE_H_ */
