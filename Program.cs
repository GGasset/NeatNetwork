using System;
using System.Collections.Generic;
using NeatNetwork.Libraries;
using NeatNetwork.Groupings;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static NeatNetwork.NetworkFiles.NeuronHolder;

namespace NeatNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            // NetworkGroup Demonstration
            // Initialize NetworkGroup
            /*NetworkGroup n = new NetworkGroup(1, 2, .1);
            List<AgroupatedNetwork> ns = new List<AgroupatedNetwork>()
            {
                new AgroupatedNetwork(new RNN(new int[] {4, 15, 20 }, new NeuronTypes[] { NeuronTypes.LSTM, NeuronTypes.LSTM }, Activation.ActivationFunctions.Sigmoid)),
                new AgroupatedNetwork(new RNN(new int[] {7, 10, 20}, new NeuronTypes[] { NeuronTypes.LSTM, NeuronTypes.LSTM }, Activation.ActivationFunctions.Sigmoid)),
                new AgroupatedNetwork(new RNN(new int[] {6, 10, 20}, new NeuronTypes[] { NeuronTypes.LSTM, NeuronTypes.LSTM }, Activation.ActivationFunctions.Sigmoid)),
            };
            List<int> executionOrder = new List<int>()
            {
                0, 1, 2,
            };
            n.Networks = ns;
            n.ExecutionOrder = executionOrder;

            // Connect network 0 to input
            n.Connect(0, Range.WholeRange, -1, Range.WholeRange);

            // Connect network 1 to network 0
            n.Connect(1, Range.WholeRange, 0, new Range(0, 15));

            // Connect network 2 to network 1
            n.Connect(2, new Range(1, 4), 1, Range.WholeRange);

            // Connect network 2 to input
            n.Connect(2, Range.WholeRange, -1, Range.WholeRange);

            n.ConnectToOutput(1, new Range(0, 15), new Range(0, 0));
            n.ConnectToOutput(2, Range.WholeRange, Range.WholeRange);*/

            // NetworkGroup Supervised learning
            /*List<double[]> X = new List<double[]>()
            {
                new double[] {15},
                new double[] {30}
            };

            List<double[]> y = new List<double[]>()
            {
                new double[] {25, 35},
                new double[] {55, 68 }
            };

            for (int i = 0; i < 100000; i++)
            {
                n.TrainBySupervisedLearning(X, y, Cost.CostFunctions.SquaredMean);
                var firstOutput = n.Execute(X[0]);
                var secondOutput = n.Execute(X[1]);

                double firstExecutionFirstError = firstOutput[0] - y[0][0];
                double firstExecutionSecondError = firstOutput[1] - y[0][1];
                double secondExecutionFirstError = secondOutput[0] - y[1][0];
                double secondExecutionSecondError = secondOutput[0] - y[1][1];

                Console.WriteLine($"{i}\nFirst output was off by:" +
                    $"{firstExecutionFirstError} and {firstExecutionSecondError}" +
                    $"\nSecond output was off by:" +
                    $"{secondExecutionFirstError} and {secondExecutionSecondError}" +
                    "\n======================");
            }*/


            //RNN reinforcement learning
            /*RNN rNN = new RNN(new int[] { 1, 13, 5, 1 }, new NeuronTypes[] { NeuronTypes.LSTM, NeuronTypes.LSTM, NeuronTypes.LSTM }, Activation.ActivationFunctions.Sigmoid);
            ReinforcementLearningRNN n = new ReinforcementLearningRNN(rNN);
            var X = new double[] { 5 };
            var expectedFirstValue = 0.5;
            for (int i = 0; i < int.MaxValue; i++)
            {
                var firstOutput = n.Execute(X)[0];
                var distanceToTarget = Math.Abs(firstOutput - expectedFirstValue + 1);

                var firstReward = 1 / (distanceToTarget);

                n.GiveReward(firstReward);

                var secondOutput = n.Execute(X)[0];
                n.GiveReward(secondOutput - firstOutput);

                n.TerminateAgent();

                Console.WriteLine($"{firstOutput}\n{secondOutput}\n----------------");
            }*/

            /*RNN n = new RNN(new int[] { 1, 13, 5, 1 }, new NeuronTypes[] { NeuronTypes.LSTM, NeuronTypes.LSTM, NeuronTypes.LSTM }, Activation.ActivationFunctions.Sigmoid);
            List<List<double[]>> X = new List<List<double[]>>()
            {
                new List<double[]>()
                {
                    new double[]
                    {
                        10,
                    },
                    new double[]
                    {
                        10
                    }
                },
            };

            List<List<double[]>> y = new List<List<double[]>>()
            {
                new List<double[]>()
                {
                    new double[]
                    {
                        0.5
                    },
                    new double[]
                    {
                        0.7
                    }
                },
            };

            for (int i = 0; i < 500000; i++)
            {
                n.SupervisedLearningBatch(X, y, 1, Cost.CostFunctions.SquaredMean, .5);
                Console.WriteLine(n.Execute(X[0][0])[0]);
                Console.WriteLine(n.Execute(X[0][1])[0]);
                Console.WriteLine("---------------------");
            }

            n.DeleteMemory();

            Console.ReadLine();*/


            /*List<double[]> X = new List<double[]>()
            {----------
                             new double[] { 0,0 },
                new double[] { 1,0 },
                new double[] { 0,1 },
                new double[] { 1,1 },
            };

            List<double[]> y = new List<double[]>()
            {
                new double[] { 1 },
                new double[] { -1 },
                new double[] { -1 },
                new double[] { 1 },
            };


           // NN Evolution Learning demonstration
           NNEvolutionManager world = new NNEvolutionManager(15000, new int[] { 2, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, 0.37, 1, .25);


           for (int i = 0; i < 4000; i++)
           {
               int j = 0;
               while (!world.AreAllNetworksScored())
               {
                   NN cn = world.GetNextToScoreNetwork();
                   double meanCost = 0;
                   for (int k = 0; k < X.Count; k++)
                   {
                       meanCost += Cost.GetCost(cn.Execute(X[k]), y[k], Cost.CostFunctions.SquaredMean);
                   }
                   meanCost /= X.Count;

                   if (i != 0 && j != 0)
                       world.SetNextNetworkToBeScoredScore(-meanCost);
                   else
                       world.SetFirstNetworkScore(-meanCost);

                   /*double score = cn.Execute(X[1])[0];
                   world.SetNextNetworkToBeScoredScore(score);*
                   j++;
               }
               world.DropWorstNetworks(.5);
               Console.WriteLine($"Generation: {i}   NetworkCount: {world.Networks.Count}   MaxScore: {world.MaxScore}");

               world.HaveChild(1, 2);
           }
           Console.WriteLine("\n");
           NN n = world.GetMaxScoredNetwork();
           for (int i = 0; i < X.Count; i++)
           {
               double output = n.Execute(X[i])[0];
               double expected = y[i][0];
               Console.WriteLine($"Output: {output}, Expected: {expected}");
           }*/

            // NN Supervised learning demonstration
            NN n = new NN(new int[] { 4, 5, 6, 5, 4, 3, 1 }, Activation.ActivationFunctions.Sigmoid, 1.5, -1.5, .5);
            List<double[]> X = new List<double[]>()
            {
                new double[] { 3, 7, 8, 9 },
            };

            double targetVal = 0.63541524;
            List<double[]> y = new List<double[]>()
            {
                new double[] { targetVal },
            };

            double learningRate = .75;
            for (int j = 0; j < 4000; j++)
            {
                n.SupervisedLearningBatch(X, y, 1, Cost.CostFunctions.SquaredMean, learningRate);
                double output;
                Console.WriteLine(output = n.Execute(X[0])[0]);
                if (targetVal == output)
                {
                    Console.WriteLine($"Overfitted in {j} steps");
                    Console.ReadKey();
                    break;
                }
            }

            n = new NN(n.ToString());

            Console.WriteLine($"{n.Execute(X[0])[0]}");
            Console.ReadKey();

            // NN Reinforcement Learning demonstration
            /*double learningRate = 1;
            ReinforcementLearningNN agent = new ReinforcementLearningNN(new NN(new int[] { 1, 15, 4, 400, 1 }, Activation.ActivationFunctions.Sigmoid), learningRate);
            double[] input = { 30 };
            double biggestOutput = 0;
            for (int i = 0; i < 5000; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    double output = agent.Execute(input)[0];
                    Console.WriteLine(output);
                    if (i != 0)
                    {
                        agent.GiveReward(output - biggestOutput);
                        biggestOutput += (output - biggestOutput) * Convert.ToInt32(output > biggestOutput);
                    }
                    else
                        biggestOutput = output;
                }
                agent.TerminateAgent();
            }*/
        }
    }
}
