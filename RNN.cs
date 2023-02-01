using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork
{
    public class RNN
    {
        public int Length => Neurons.Count;
        public readonly int InputLength;
        public int OutputLength => Neurons[Neurons.Count - 1].Count;

        /// <summary>
        /// Doesn't include input
        /// </summary>
        public int[] Shape => GetShape();

        internal ActivationFunctions ActivationFunction;
        internal List<List<NeuronHolder>> Neurons;

        internal double InitialMaxMutationValue;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxWeight;
        internal double MinWeight;
        internal double WeightClosestTo0;
        internal double NewBiasValue;
        internal double NewNeuronChance;
        internal double NewLayerChance;
        internal double FieldMaxMutation;
        internal double MaxMutationOfFieldMaxMutation;
        internal double MaxMutationOfMutationValueOfFieldMaxMutation;
        internal double MutationChance;

        /// <summary>
        ///
        /// </summary>
        /// <param name="shape">Includes input layer</param>
        /// <param name="layerTypes">Doesn't include input layer</param>
        /// <param name="activationFunction"></param>
        /// <param name="startingBias"></param>
        /// <param name="minWeight"></param>
        /// <param name="maxWeight"></param>
        /// <param name="weightClosestTo0"></param>
        public RNN(int[] shape, NeuronHolder.NeuronTypes[] layerTypes, ActivationFunctions activationFunction, double startingBias = 1, double minWeight = -1.5, double maxWeight = 1.5, double weightClosestTo0 = .37,
            double newNeuronChance = .05, double newLayerChance = .02, double mutationChance = .2,
            double initialMaxMutationGridValue = .5, double fieldMaxMutation = .07, double maxMutationOfFieldMaxMutation = .03, double maxMutationOfMutationValueOfFieldMaxMutation = .01)
        {
            ActivationFunction = activationFunction;
            Neurons = new List<List<NeuronHolder>>();
            MaxMutationGrid = new List<List<double>>();
            for (int i = 1; i < shape.Length; i++)
            {
                (List<NeuronHolder> layerNeurons, List<double> layerMaxMutations) = InstantiateLayer(layerTypes[i - 1], i, shape[i], shape[i - 1], initialMaxMutationGridValue,
                    startingBias, minWeight, weightClosestTo0, maxWeight);
                Neurons.Add(layerNeurons);
                MaxMutationGrid.Add(layerMaxMutations);
            }

            InputLength = shape[0];

            NewBiasValue = startingBias;
            MaxWeight = maxWeight;
            MinWeight = minWeight;
            WeightClosestTo0 = weightClosestTo0;

            NewNeuronChance = newNeuronChance;
            NewLayerChance = newLayerChance;
            MutationChance = mutationChance;
            InitialMaxMutationValue = initialMaxMutationGridValue;
            FieldMaxMutation = fieldMaxMutation;
            MaxMutationOfFieldMaxMutation = maxMutationOfFieldMaxMutation;
            MaxMutationOfMutationValueOfFieldMaxMutation = maxMutationOfMutationValueOfFieldMaxMutation;
        }

        private class AsyncLayerInstatiator
        {
            private readonly NeuronHolder.NeuronTypes LayerType;
            private readonly int LayerIndex;
            private readonly int LayerLength;
            private readonly int PreviousLayerLength;

            internal AsyncLayerInstatiator(NeuronHolder.NeuronTypes layerType, int layerIndex, int layerLength, int previousLayerLength)
            {
                LayerType = layerType;
                LayerIndex = layerIndex;
                LayerLength = layerLength;
                PreviousLayerLength = previousLayerLength;
            }

            internal Task<(List<NeuronHolder> neurons, List<double> layerMaxMutation)> InstatiateLayerAsync(double initialMaxMutationValue, double bias, double minWeight, double weightClosestTo0, double maxWeight)
                => Task.Run(() => InstantiateLayer(LayerType, LayerIndex, LayerLength, PreviousLayerLength, initialMaxMutationValue, bias, minWeight, weightClosestTo0, maxWeight));
        }

        private static (List<NeuronHolder>, List<double> layerMaxMutations) InstantiateLayer(NeuronHolder.NeuronTypes neuronType, int layerIndex, int layerLength, int previousLayerLength, double initialMaxMutationValue, 
            double bias, double minWeigth, double weightClosestTo0, double maxWeight)
        {
            List<Task<NeuronHolder>> neuronTasks = new List<Task<NeuronHolder>>();
            for (int i = 0; i < layerLength; i++)
                neuronTasks.Add(Task.Run(() => new NeuronHolder(neuronType, layerIndex, previousLayerLength, bias, maxWeight, minWeigth, weightClosestTo0)));

            int maxMutationPartitionLength = 3000;
            int maxMutationPartitions = layerLength / maxMutationPartitionLength;
            int lastMaxMutationPartitionLength = layerLength % maxMutationPartitionLength;

            List<Task<List<double>>> maxMutationPartitionTasks = new List<Task<List<double>>>();
            for (int i = 0; i < maxMutationPartitions; i++)
            {
                maxMutationPartitionTasks.Add(Task.Run(() => InstantiateMaxMutationList(initialMaxMutationValue, maxMutationPartitionLength)));
            }
            maxMutationPartitionTasks.Add(Task.Run(() => InstantiateMaxMutationList(initialMaxMutationValue, lastMaxMutationPartitionLength)));

            foreach (var neuronTask in neuronTasks)
            {
                neuronTask.Wait();
            }
            foreach (var maxMutationListPartitionTask in maxMutationPartitionTasks)
            {
                maxMutationListPartitionTask.Wait();
            }

            List<NeuronHolder> neurons = new List<NeuronHolder>();
            foreach (var neuronTask in neuronTasks)
            {
                neurons.Add(neuronTask.Result);
            }

            List<double> layerMaxMutations = new List<double>();
            foreach (var maxMutationListPartitionTask in maxMutationPartitionTasks)
            {
                layerMaxMutations.AddRange(maxMutationListPartitionTask.Result);
            }

            return (neurons, layerMaxMutations);
        }

        private static List<double> InstantiateMaxMutationList(double maxMutationValue, int listLength)
        {
            var list = new List<double>();
            for (int i = 0; i < listLength; i++)
            {
                list.Add(maxMutationValue);
            }
            return list;
        }

        public double[] Execute(double[] input) => Execute(input, out _, out _);

        /// <summary>
        ///
        /// </summary>
        /// <param name="input"></param>
        /// <param name="networkExecutionValues"></param>
        /// <param name="neuronActivations">includes input</param>
        /// <returns></returns>
        public double[] Execute(double[] input, out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronActivations)
        {
            networkExecutionValues = new List<NeuronExecutionValues[]>();
            neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i < Length; i++)
            {
                (double[] layerOutput, NeuronExecutionValues[] layerExecutionValues) = ExecuteLayer(i, neuronActivations);

                neuronActivations.Add(layerOutput);
                networkExecutionValues.Add(layerExecutionValues);
            }

            return neuronActivations[Length];
        }

        private (double[] output, NeuronExecutionValues[] executionValues) ExecuteLayer(int i, List<double[]> neuronActivations)
        {
            int layerLength = Neurons[i].Count;
            Task<(double, NeuronExecutionValues)>[] tasks = new Task<(double, NeuronExecutionValues)>[layerLength];
            AsyncNeuronExecutor[] asyncExecutors = new AsyncNeuronExecutor[layerLength];

            for (int j = 0; j < layerLength; j++)
            {
                asyncExecutors[j] = new AsyncNeuronExecutor(neuronActivations, Neurons[i][j]);
                tasks[j] = asyncExecutors[j].ExecuteAsync(ActivationFunction);
            }

            foreach (var task in tasks)
            {
                task.Wait();
            }
            
            double[] output = new double[layerLength];
            NeuronExecutionValues[] executionValues = new NeuronExecutionValues[layerLength];

            for (int j = 0; j < layerLength; j++)
                (output[j], executionValues[j]) = tasks[j].Result;

            return (output, executionValues);
        }

        private class AsyncNeuronExecutor
        {
            private readonly List<double[]> neuronActivations;
            private readonly NeuronHolder neuron;

            internal AsyncNeuronExecutor(List<double[]> neuronActivations, NeuronHolder neuron)
            {
                this.neuronActivations = neuronActivations;
                this.neuron = neuron;
            }

            internal Task<(double output, NeuronExecutionValues executionValues)> ExecuteAsync(ActivationFunctions activationFunction) =>
                Task.Run(() => neuron.Execute(neuronActivations, activationFunction));
        }

        /// <summary>
        /// Used for auto encoder networks
        /// </summary>
        /// <param name="layerI">layerI is inclusive and doesn't include input layer</param>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] ExecuteUpToLayer(int layerI, double[] input)
        {
            if (layerI >= Length || Neurons[layerI].Count != input.Length) throw new ArgumentOutOfRangeException();
            
            var neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i <= layerI; i++)
            {
                (double[] layerOutput, _) = ExecuteLayer(i, neuronActivations);

                neuronActivations.Add(layerOutput);
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        /// <summary>
        /// Used for auto encoder networks
        /// </summary>
        /// <param name="layerI">doesn't include input layer</param>
        /// <param name="layerActivations"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException">input length must equal layer layerI length</exception>
        public double[] ExecuteFromLayer(int layerI, double[] layerActivations)
        {
            if (Neurons[layerI].Count != layerActivations.Length) throw new ArgumentOutOfRangeException("input length doesn't match layer length");

            var neuronActivations = new List<double[]>()
            {
                new double[InputLength]
            };

            for (int i = 0; i < layerI; i++)
            {
                int layerLength = Neurons[i].Count;
                neuronActivations.Add(new double[layerLength]);
            }

            neuronActivations.Add(layerActivations);

            for (int i = layerI + 1; i < Length; i++)
            {
                (double[] layerOutput, _) = ExecuteLayer(i, neuronActivations);
                neuronActivations.Add(layerOutput);
            }
            return neuronActivations[Length];
        }

        #region Gradient Learning

        /// <summary>
        /// 
        /// </summary>
        /// <param name="X"></param>
        /// <param name="y"></param>
        /// <param name="costFunction"></param>
        /// <param name="learningRate"></param>
        /// <param name="testSize"></param>
        /// <param name="batchLength">beware that it consumes a lot of memory as this number increases</param>
        /// <param name="shuffleData"></param>
        /// <returns>Test cost</returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="ArgumentOutOfRangeException">Test size must be between 0 and 1</exception>
        public double SupervisedTrain(List<List<double[]>> X, List<List<double[]>> y, Cost.CostFunctions costFunction, double learningRate, double testSize = 0.15, int batchLength = 350, bool shuffleData = true)
        {
            if (X.Count != y.Count) throw new ArgumentException("X.Count doesn't equal y.Count");
            if (testSize > 1 || testSize < 0) throw new ArgumentOutOfRangeException("Test size doesn't fall between 0 and 1");

            if (shuffleData)
                (X, y) = DataManipulation.ShuffleData(X, y);

            ((List<List<double[]>> trainX, List<List<double[]>> trainY), (List<List<double[]>> testX, List<List<double[]>> testY)) = DataManipulation.SliceData(X, y, 1 - testSize);
            
            int batchCount = trainX.Count / batchLength;
            int lastBatchLength = trainX.Count % batchLength;

            // Training
            DeleteMemory();
            for (int i = 0; i < batchCount; i++)
            {
                SupervisedLearningBatch(trainX, trainY, costFunction, learningRate, i * batchLength, batchLength);
            }
            SupervisedLearningBatch(trainX, trainY, costFunction, learningRate, batchCount * batchLength, lastBatchLength);

            //Testing
            double meanCost = 0;
            int totalTimeSteps = 0;
            for (int i = 0; i < testX.Count; i++)
            {
                for (int j = 0; j < testX[i].Count; j++)
                {
                    double[] currentOutput = Execute(testX[i][j]);
                    double currentCost = Cost.GetCost(currentOutput, testY[i][j], costFunction);
                    meanCost += currentCost;
                    totalTimeSteps++;
                }
                DeleteMemory();
            }
            meanCost /= totalTimeSteps;
            return meanCost;
        }

        public void SupervisedLearningBatch(List<List<double[]>> X, List<List<double[]>> y, Cost.CostFunctions costFunction, double learningRate, int startingIndex, int batchLength)
        {
            AsyncGradientCalculator[] gradientCalculators = new AsyncGradientCalculator[batchLength];
            List<Task<List<List<NeuronHolder>>>> gradientTasks = new List<Task<List<List<NeuronHolder>>>>();
            for (int i = 0; i < batchLength; i++)
            {
                gradientCalculators[i] = new AsyncGradientCalculator(X[i + startingIndex], y[i + startingIndex], this);
                gradientTasks.Add(gradientCalculators[i].GetGradientsAsync(costFunction));
            }

            foreach (var task in gradientTasks)
            {
                task.Wait();
            }

            foreach (var gradientTask in gradientTasks)
                SubtractGrads(gradientTask.Result, learningRate);
        }

        private class AsyncGradientCalculator
        {
            private RNN n;
            private readonly List<double[]> X, y;

            internal AsyncGradientCalculator(List<double[]> x, List<double[]> y, RNN n)
            {
                X = x;
                this.y = y;
                this.n = n;
            }

            internal Task<List<List<NeuronHolder>>> GetGradientsAsync(Cost.CostFunctions costFunction)
            {
                n = n.Clone();
                return Task.Run(() => n.GetSupervisedLearningGradients(X, y, costFunction, false));
            }
        }

        public void SupervisedLearningBatch(List<List<double[]>> X, List<List<double[]>> y, double batchSize, Cost.CostFunctions costFunction, double learningRate)
        {
            List<List<List<NeuronHolder>>> gradients = new List<List<List<NeuronHolder>>>();
            X = new List<List<double[]>>(X.ToArray());
            y = new List<List<double[]>>(y.ToArray());

            batchSize = Math.Abs(batchSize);
            batchSize *= 1 * Convert.ToInt16(batchSize > 1) + X.Count * Convert.ToInt16(batchSize <= 1);
            batchSize = Math.Ceiling(batchSize);

            DeleteMemory();
            Random r = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < batchSize; i++)
            {
                int dataI = r.Next(X.Count);
                gradients.Add(GetSupervisedLearningGradients(X[dataI], y[dataI], costFunction, false));

                X.RemoveAt(dataI);
                y.RemoveAt(dataI);

                DeleteMemory();
            }

            SubtractGrads(gradients, learningRate);
        }

        public void TrainBySupervisedLearning(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, double learningRate) =>
            SubtractGrads(GetSupervisedLearningGradients(X, y, costFunction), learningRate);

        internal List<List<NeuronHolder>> GetSupervisedLearningGradients(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, bool deleteMemoryBeforeAndAfter = true) =>
            GetSupervisedLearningGradients(X, y, costFunction, out _, deleteMemoryBeforeAndAfter);

        internal List<List<NeuronHolder>> GetSupervisedLearningGradients(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, out List<List<double>> inputGradients, bool deleteMemoryBeforeAndAfter = true)
        {
            List<double[]> outputs = new List<double[]>();
            List<List<NeuronExecutionValues[]>> networkExecutionsValues = new List<List<NeuronExecutionValues[]>>();
            List<List<double[]>> networkExecutionsNeuronOutputs = new List<List<double[]>>();
            List<double[]> costGradients = new List<double[]>();

            if (deleteMemoryBeforeAndAfter)
                DeleteMemory();

            int tSCount = X.Count;
            for (int i = 0; i < tSCount; i++)
            {
                double[] currentNetworkOutput;
                outputs.Add(currentNetworkOutput = Execute(X[i], out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronOutputs));

                networkExecutionsValues.Add(networkExecutionValues);
                networkExecutionsNeuronOutputs.Add(neuronOutputs);

                costGradients.Add(Derivatives.DerivativeOf(currentNetworkOutput, y[i], costFunction));
            }

            var output = GetGradients(costGradients, networkExecutionsValues, networkExecutionsNeuronOutputs, out inputGradients);

            if (deleteMemoryBeforeAndAfter)
                DeleteMemory();

            return output;
        }

        /// <param name="executionValues">3D Grid in which time is the highest dimension and then layersStrs and finally neurons</param>
        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations) =>
            GetGradients(costGradients, executionValues, neuronActivations, out _);

        /// <summary>
        ///
        /// </summary>
        /// <param name="costGradients"></param>
        /// <param name="executionValues">3D Grid in which time is the highest dimension and then layersStrs and finally neurons</param>
        /// <param name="neuronActivations"></param>
        /// <param name="inputGradients">List that represents time containing input costs</param>
        /// <returns></returns>
        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations, out List<List<double>> inputGradients)
        {
            List<List<NeuronHolder>> output = new List<List<NeuronHolder>>();
            List<List<List<double>>> neuronOutputGradientsGrid = ValueGeneration.GetTemporalNetworkCostGrid(costGradients, InputLength, Shape);
            int tSCount = costGradients.Count;
            int lastLayerI = Neurons.Count - 1;

            for (int layerI = lastLayerI; layerI >= 0; layerI--)
            {
                output.Add(new List<NeuronHolder>());
                for (int neuronI = 0; neuronI < Neurons[layerI].Count; neuronI++)
                {
                    List<NeuronExecutionValues> neuronExecutionValues = new List<NeuronExecutionValues>();
                    for (int t = 0; t < tSCount; t++)
                        neuronExecutionValues.Add(executionValues[t][layerI][neuronI]);

                    NeuronHolder cNeuron = Neurons[layerI][neuronI];

                    output[lastLayerI - layerI].Add(cNeuron.GetGradients(neuronOutputGradientsGrid[layerI + 1][neuronI], neuronActivations, neuronExecutionValues, ActivationFunction, out List<double[]> connectionsGradients));

                    // Update neuronOutputGradientsGrid
                    for (int t = 0; t < tSCount; t++)
                        for (int i = 0; i < cNeuron.Connections.Length; i++)
                        {
                            Point connectedNeuronPos = cNeuron.Connections.ConnectedNeuronsPos[i];
                            neuronOutputGradientsGrid[connectedNeuronPos.X][connectedNeuronPos.Y][t] -= connectionsGradients[t][i];
                        }
                }
            }
            output.Reverse();
            inputGradients = neuronOutputGradientsGrid[0];
            return output;
        }

        internal void SubtractGrads(List<List<List<NeuronHolder>>> gradients, double learningRate)
        {
            foreach (var networkGradients in gradients)
                SubtractGrads(networkGradients, learningRate);
        }

        internal void SubtractGrads(List<List<NeuronHolder>> gradients, double learningRate)
        {
            for (int i = 0; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j], learningRate);
        }

        #endregion Gradient Learning

        #region Evolution learning

        public void AddInputNeuron()
        {
            int previousInputLength = InputLength;
            for (int i = 0; i < Neurons[0].Count; i++)
                Neurons[0][i].AddConnection(0, previousInputLength, ValueGeneration.GenerateWeight(MinWeight, MaxWeight, WeightClosestTo0));
        }

        #endregion Evolution learning

        /// <summary>
        /// For proper training use only after each train step, it isn't neccesary to use after each train step
        /// </summary>
        public void DeleteMemory()
        {
            for (int i = 0; i < Length; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].DeleteMemory();
        }

        public override string ToString()
        {
            string output = string.Empty;
            foreach (var layer in Neurons)
            {
                foreach (var neuron in layer)
                {
                    output += neuron.ToString() + "/";
                }
                output = output.Remove(output.LastIndexOf('/'));
                output += "\n";
            }
            output = output.Remove(output.LastIndexOf('\n'));
            return output;
        }

        public RNN(string str)
        {
            Neurons = new List<List<NeuronHolder>>();
            string[] layersStrs = str.Split('\n');

            for (int i = 0; i < layersStrs.Length; i++)
            {
                Neurons.Add(new List<NeuronHolder>());

                string[] neuronsStrs = layersStrs[i].Split('/');
                foreach (var neuronStr in neuronsStrs)
                    Neurons[i].Add(new NeuronHolder(neuronStr));
            }

            InputLength = Neurons[0][0].Connections.Length;
        }

        private int[] GetShape()
        {
            int[] shape = new int[Length];
            for (int i = 0; i < Neurons.Count; i++)
                shape[i] = Neurons[i].Count;
            return shape;
        }

        public RNN Clone()
        {
            RNN output = (RNN)MemberwiseClone();
            output.Neurons = new List<List<NeuronHolder>>();
            output.MaxMutationGrid = MaxMutationGrid;
            for (int i = 0; i < Neurons.Count; i++)
            {
                output.Neurons.Add(new List<NeuronHolder>());
                output.MaxMutationGrid.Add(new List<double>());
                for (int j = 0; j < Neurons[i].Count; j++)
                {
                    output.Neurons[i].Add(Neurons[i][j].Clone());
                    output.MaxMutationGrid[i].Add(MaxMutationGrid[i][j]);
                }
            }
            return output;
        } 
    }
}