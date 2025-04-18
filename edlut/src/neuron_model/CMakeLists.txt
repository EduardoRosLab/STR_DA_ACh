target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModel.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModelVector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressNeuronModelTable.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressSynchronousTableBasedModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CompressTableBasedModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CurrentSynapseModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EgidioGranuleCell_TimeDriven.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EventDrivenInputDevice.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EventDrivenNeuronModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/HHTimeDrivenModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InputCurrentNeuronModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/InputSpikeNeuronModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichTimeDrivenModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_IS.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_DA.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/AsabukiNeuron.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelTable.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PoissonGeneratorDeviceVector.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/SinCurrentDeviceVector.cpp
	#	${CMAKE_CURRENT_SOURCE_DIR}/SRMTableBasedModel.cpp
        #	${CMAKE_CURRENT_SOURCE_DIR}/SRMTimeDrivenModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SynchronousTableBasedModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TableBasedModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenInferiorOliveCell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenInputDevice.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenNeuronModel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell_IP.cpp
        #	${CMAKE_CURRENT_SOURCE_DIR}/VectorBufferedState.cpp
        #	${CMAKE_CURRENT_SOURCE_DIR}/VectorSRMState.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/VectorNeuronState.cpp)

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/AdExTimeDrivenModel_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/EgidioGranuleCell_TimeDriven_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/HHTimeDrivenModel_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/IzhikevichTimeDrivenModel_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/LIFTimeDrivenModel_IS_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenNeuronModel_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/TimeDrivenPurkinjeCell_GPU_C_INTERFACE.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/VectorNeuronState_GPU_C_INTERFACE.cu
            )
endif()

if (CUDA_FOUND)
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelFactory.cu
            )
else()
    target_sources(${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/NeuronModelFactory.cpp
            )
endif()
