target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModelVector.h
	${CMAKE_CURRENT_SOURCE_DIR}/CompressNeuronModelTable.h
	${CMAKE_CURRENT_SOURCE_DIR}/CompressSynchronousTableBasedModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/CompressTableBasedModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/CurrentSynapseModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/EgidioGranuleCell_TimeDriven.h
	${CMAKE_CURRENT_SOURCE_DIR}/EventDrivenInputDevice.h
	${CMAKE_CURRENT_SOURCE_DIR}/EventDrivenNeuronModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/HHTimeDrivenModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/InputCurrentNeuronModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/InputSpikeNeuronModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichTimeDrivenModel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_IS.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_DA.h
        ${CMAKE_CURRENT_SOURCE_DIR}/AsabukiNeuron.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModel.h
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelFactory.h
	${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelTable.h
	${CMAKE_CURRENT_SOURCE_DIR}/PoissonGeneratorDeviceVector.h
	${CMAKE_CURRENT_SOURCE_DIR}/SinCurrentDeviceVector.h
#	${CMAKE_CURRENT_SOURCE_DIR}/SRMTableBasedModel.h
#	${CMAKE_CURRENT_SOURCE_DIR}/SRMTimeDrivenModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/SynchronousTableBasedModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/TableBasedModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenInferiorOliveCell.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenInputDevice.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenNeuronModel.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell.h
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell_IP.h
#	${CMAKE_CURRENT_SOURCE_DIR}/VectorBufferedState.h
#	${CMAKE_CURRENT_SOURCE_DIR}/VectorSRMState.h
        ${CMAKE_CURRENT_SOURCE_DIR}/VectorNeuronState.h)

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModel_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModel_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/EgidioGranuleCell_TimeDriven_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/HHTimeDrivenModel_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/HHTimeDrivenModel_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichTimeDrivenModel_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichTimeDrivenModel_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_IS_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_IS_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenNeuronModel_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenNeuronModel_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell_GPU2.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/VectorNeuronState_GPU_C_INTERFACE.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/VectorNeuronState_GPU2.cuh
            )
endif()
