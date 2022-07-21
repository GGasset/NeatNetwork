using System;
using System.Collections.Generic;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    internal class NNEvolutionManager
    {
        internal List<NN> Networks;
        internal List<double> Scores { get; private set; }
        internal double MaxScore { get; private set; }
        internal double MinScore { get; private set; }
        internal int MaxScoredNetwork { get; private set; }

        internal NNEvolutionManager(int startingNetworkCount, int[] layerLengths, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1,
            double mutationChance = .1, double fieldMaxMutation = .04, double initialMaxMutationValue = .27, double newNeuronChance = .2, double newLayerChance = .05,
            double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            MaxScore = double.MinValue;
            MinScore = double.MaxValue;

            Networks = new List<NN>();
            Scores = new List<double>();
            for (int i = 0; i < startingNetworkCount; i++)
            {
                Networks.Add(new NN(layerLengths, activation, maxWeight, minWeight, weightClosestTo0, startingBias, mutationChance, fieldMaxMutation, initialMaxMutationValue, newNeuronChance, newLayerChance
                    , initialValueForMaxMutation, maxMutationOfMutationValues, maxMutationOfMutationValueOfMutationValues));
            }
        }

        /// <summary>
        /// Function only applies to scored networks
        /// </summary>
        /// <param name="minChilds"></param>
        /// <param name="maxChilds"></param>
        internal void HaveChild(int minChilds, int maxChilds)
        {
            (minChilds, maxChilds) = (Math.Max(minChilds, 0), Math.Max(maxChilds, 0));

            for (int i = 0; i < Scores.Count; i++)
            {
                double score = Scores[i];
                double currentChildCount = Math.Round((score - MinScore) / (MaxScore - MinScore) * maxChilds);
                currentChildCount = Math.Max(currentChildCount, minChilds);

                for (int j = 0; j < currentChildCount; j++)
                {
                    NN n = new NN(Networks[i].ToString());
                    n.Evolve();
                    Networks.Add(n);
                }
            }
        }

        /// <summary>
        /// Don't use really large negative or positive numbers for correct functioning
        /// </summary>
        /// <param name="score">Must be positive to properly work</param>
        internal void SetNextNetworkToBeScoredScore(double score)
        {
            if (score > MaxScore)
            {
                MaxScore = score;
                MaxScoredNetwork = Scores.Count;
            }
            if (score < MinScore)
            {
                MinScore = score;
            }

            Scores.Add(score);
            /*MaxScore += (score - MaxScore) * Convert.ToInt32(score > MaxScore);
            MaxScoredNetwork += (Scores.Count - 1 - MaxScoredNetwork) * Convert.ToInt32(score > MaxScore);
            MinScore += (score - MinScore) * Convert.ToInt32(score < MinScore);*/
        }

        internal NN GetMaxScoredNetwork() => Networks[MaxScoredNetwork];

        internal NN GetNextToScoreNetwork() => GetNextToScoreNetwork(out _);

        internal NN GetNextToScoreNetwork(out int networkIndex)
        {
            networkIndex = Scores.Count;
            return Networks[Scores.Count];
        }

        internal bool AreAllNetworksScored()
        {
            return Scores.Count == Networks.Count;
        }

        /// <summary>
        /// This can be used to add other learning techniques while in network life cycle
        /// </summary>
        /// <param name="n"></param>
        /// <param name="index"></param>
        internal void SetNetwork(NN n, int index)
        {
            Networks[index] = n;
        }

        internal void SetNextToScoreNetwork(NN n)
        {
            Networks[Scores.Count] = n;
        }

        internal void SetLastScoredNetwork(NN n)
        {
            Networks[Scores.Count - 1] = n;
        }
    }
}
