/*

  Neuro.NET - Library of neural networks for .NET
  Copyright (C) 2001-2015  Alex Cherkasov. All rights reserved.
                           email: info@xpidea.com
                           web: http://xpidea.com/

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
  02110-1301, USA.

{   CREDITS:                                                    }
{   This work is based on publications of:                      }
{          -Christopher M. Bishop                               }
{          -Jose C. Principe                                    }
{          -Samuel J. Rogers                                    }
{          -Laurene V. Fausett                                  }
{          -Simon S. Haykin                                     }
  
*/


using System;
using System.IO;
using xpidea.neuro.net.adaline;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.backprop
{
    /// <summary>
    ///     Implements a node in BackPropagation network.
    /// </summary>
    public class BackPropagationNode : FeedForwardNode
    {
        /// <summary>
        ///     Overridden. Implements sigmoid tranfer function for BackProp nodes
        /// </summary>
        /// <param name="value">The node value</param>
        /// <returns>A transfer result.</returns>
        protected override double Transfer(double value)
        {
            return 1F/(1F + Math.Exp(-value)); //Sigmoid transfer 
        }
    }

    /// <summary>
    ///     Implements the link  in the Backpropagaion network.
    /// </summary>
    public class BackPropagationLink : NeuroLink
    {
        /// <summary>
        ///     Class field stores link's delta weight.
        /// </summary>
        private double linkDelta;

        /// <summary>
        ///     Constructs the link and calls <see cref="xpidea.neuro.net.backprop.BackPropagationLink.DoAfterCreate" /> method to
        ///     initialize data.
        /// </summary>
        public BackPropagationLink()
        {
            DoAfterCreate();
        }

        /// <summary>
        ///     Returns link delta.
        /// </summary>
        public double LinkDelta
        {
            get { return GetLinkDelta(); }
            set { SetLinkDelta(value); }
        }

        /// <summary>
        ///     Getter method of <see cref="xpidea.neuro.net.backprop.BackPropagationLink.GetLinkDelta" /> property.
        /// </summary>
        /// <returns>Link's delta.</returns>
        protected virtual double GetLinkDelta()
        {
            return linkDelta;
        }

        /// <summary>
        ///     Initializes link weight to a random value from -1..1  and link delta to 0.
        /// </summary>
        protected virtual void DoAfterCreate()
        {
            Weight = Random(-1F, 1F);
            LinkDelta = 0;
        }

        /// <summary>
        ///     Setter method of <see cref="xpidea.neuro.net.backprop.BackPropagationLink.LinkDelta" /> property.
        /// </summary>
        /// <param name="delta">New delta value.</param>
        protected virtual void SetLinkDelta(double delta)
        {
            linkDelta = delta;
        }

        /// <summary>
        ///     Overridden.Loads link's data from the binary stream.
        /// </summary>
        /// <param name="binaryReader">A binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            linkDelta = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden. Stores link's data in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">A binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(linkDelta);
        }

        /// <summary>
        ///     Overridden.Updates link weight according the formula.
        /// </summary>
        /// <param name="deltaWeight">Delta of link weight change.</param>
        public override void UpdateWeight(double deltaWeight)
        {
            Weight = Weight + deltaWeight + ((BackPropagationOutputNode) OutNode).Momentum*LinkDelta;
            LinkDelta = deltaWeight;
        }
    }

    /// <summary>
    ///     Represents an output node in backpropagation network.
    /// </summary>
    public class BackPropagationOutputNode : BackPropagationNode
    {
        /// <summary>
        ///     Private variable stores node's learning rate.
        /// </summary>
        private double learningRate;

        /// <summary>
        ///     Private variable stores node momentum.
        /// </summary>
        private double momentum;

        /// <summary>
        ///     Constructs BackPropagationOutputNode.
        /// </summary>
        /// <param name="learningRate">Learning rate value.</param>
        /// <param name="momentum">Momentum value.</param>
        public BackPropagationOutputNode(double learningRate, double momentum)
        {
            LearningRate = learningRate;
            Momentum = momentum;
        }

        /// <summary>
        ///     Node momentum.
        /// </summary>
        public double Momentum
        {
            get { return GetNodeMomentum(); }
            set { SetNodeMomentum(value); }
        }

        /// <summary>
        ///     Node learning rate.
        /// </summary>
        public double LearningRate
        {
            get { return GetNodeLearningRate(); }
            set { SetNodeLearningRate(value); }
        }

        /// <summary>
        ///     Method computes node error, based on current node value and previous node error.
        /// </summary>
        /// <returns>New value of error.</returns>
        protected virtual double ComputeError()
        {
            var nv = Value; //optimized for speed
            return nv*(1F - nv)*(Error - nv);
        }

        /// <summary>
        ///     Getter method of <see cref="xpidea.neuro.net.backprop.BackPropagationOutputNode.Momentum" /> property.
        /// </summary>
        /// <returns>Node momentum.</returns>
        protected virtual double GetNodeMomentum()
        {
            return momentum;
        }

        /// <summary>
        ///     Setter method of <see cref="xpidea.neuro.net.backprop.BackPropagationOutputNode.Momentum" /> property.
        /// </summary>
        /// <param name="momentum">New node momentum value.</param>
        protected virtual void SetNodeMomentum(double momentum)
        {
            this.momentum = momentum;
        }

        /// <summary>
        ///     Getter method of <see cref="xpidea.neuro.net.backprop.BackPropagationOutputNode.LearningRate" /> property.
        /// </summary>
        /// <returns>Node's learning rate.</returns>
        protected virtual double GetNodeLearningRate()
        {
            return learningRate;
        }

        /// <summary>
        ///     Setter method of <see cref="xpidea.neuro.net.backprop.BackPropagationOutputNode.LearningRate" /> property.
        /// </summary>
        /// <param name="learningRate">A new learning rate value.</param>
        protected virtual void SetNodeLearningRate(double learningRate)
        {
            this.learningRate = learningRate;
        }

        /// <summary>
        ///     Overridden.Loads node data from the binary stream.
        /// </summary>
        /// <param name="binaryReader">A binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            momentum = binaryReader.ReadDouble();
            learningRate = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Stores node data in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">A binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(momentum);
            binaryWriter.Write(learningRate);
        }

        /// <summary>
        ///     Overridden.Makes the node to learn a new pattern data.
        /// </summary>
        public override void Learn()
        {
            var err = ComputeError();
            Error = err;
            for (var i = 0; i < InLinks.Count; i++)
            {
                var delta = LearningRate*err*InLinks[i].InNode.Value; //(2.18)
                InLinks[i].UpdateWeight(delta);
            }
        }
    }


    /// <summary>
    ///     Represents a node in middle layer(s) of backpropagation network.
    /// </summary>
    public class BackPropagationMiddleNode : BackPropagationOutputNode
    {
        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        public BackPropagationMiddleNode(double learningRate, double momentum) : base(learningRate, momentum)
        {
        }

        /// <summary>
        ///     Overridden.Computes node error.
        /// </summary>
        /// <returns>New node error.</returns>
        protected override double ComputeError()
        {
            double total = 0F;
            foreach (var link in OutLinks)
                total += link.WeightedOutError();
            var nv = Value;
            return nv*(1F - nv)*total;
        }
    }

    /// <summary>
    ///     Class implementing Backpropagation network.
    /// </summary>
    /// <remarks>
    ///     The backpropagation technique is basically an extension of the adaline. The backpropagation network is a multilayer
    ///     perceptron model with an input layer, one or more hidden layers, and an output layer. These layers are organized
    ///     into interconnected layers, allowing the backpropagation network to escape the adaline's linear separability
    ///     limitations making this network much more powerful. This approach was documented by Werbos (1974).
    ///     The backpropagation network is used for problems that involve classification, projection, interpretation, and
    ///     generalization. The nodes in the backpropagation neural network are interconnected via weighted links with each
    ///     node usually connecting to the next layer up till the output layer which provides output for the network. The input
    ///     patterns values are presented and assigned to the input nodes of the input layer. The input values are initialized
    ///     to values between -1.1 and 1.1. The nodes in the next layer receive the input values through links and compute
    ///     output values of their own, which are then passed to the next layer. These values propagate forward through the
    ///     layers till the output layer is reached, or put another way, till each output layer node has produced an output
    ///     value for the network. The desired output for the input pattern is used to compute an error value for each node in
    ///     the output layer, and then propagated backwards (and here's where the network name comes in) through the network as
    ///     the delta rule is used to adjust the link values to produce better the desired output. Once the error produced by
    ///     the patterns in the training set is below a given tolerance, the training is complete and the network is presented
    ///     new input patterns and produce an output based on the experience it gained from the learning process.
    ///     <img src="Backprop.jpg"></img>
    /// </remarks>
    public class BackPropagationNetwork : AdalineNetwork
    {
        /// <summary>
        ///     Stores network momentum.
        /// </summary>
        private readonly double momentum;

        /// <summary>
        ///     Is a index of first middle layer node in the <see cref="xpidea.neuro.net.NeuralNetwork.nodes" /> array.
        /// </summary>
        protected int firstMiddleNode;

        /// <summary>
        ///     Is a index of first output node in the <see cref="xpidea.neuro.net.NeuralNetwork.nodes" /> array.
        /// </summary>
        protected int firstOutputNode;

        /// <summary>
        ///     Variable stores total number of layers in the network.
        /// </summary>
        private int layersCount;

        /// <summary>
        ///     Array definening number of nodes in each layer of the network.
        /// </summary>
        private int[] nodesInLayer;

        /// <summary>
        ///     Constructs the BackPropagation network.
        /// </summary>
        /// <param name="learningRate">Leraning rate of the network.</param>
        /// <param name="momentum">Momentum</param>
        /// <param name="nodesInEachLayer">Array of integers specifying number of nodes in each layer of the network.</param>
        public BackPropagationNetwork(double learningRate, double momentum, int[] nodesInEachLayer)
        {
            nodesCount = 0;
            linksCount = 0;
            layersCount = nodesInEachLayer.Length;
            nodesInLayer = new int[layersCount];
            for (var i = 0; i < layersCount; i++)
            {
                nodesInLayer[i] = nodesInEachLayer[i];
                nodesCount += nodesInLayer[i];
                if (i > 0)
                    linksCount += nodesInLayer[i - 1]*nodesInLayer[i];
            }
            this.learningRate = learningRate;
            this.momentum = momentum;
            CreateNetwork();
        }

        /// <summary>
        ///     Creates not initialized instance of BackPropagation network.
        /// </summary>
        public BackPropagationNetwork()
        {
            nodesInLayer = null;
        }

        /// <summary>
        ///     Creates the network from a file.
        /// </summary>
        public BackPropagationNetwork(string fileName) : base(fileName)
        {
        }

        private int GetNodesInLayer(int index)
        {
            return nodesInLayer[index];
        }

        private NeuroNode GetMiddleNode(int index)
        {
            if ((index >= GetMiddleNodesCount()) || (index < 0))
                throw new ENeuroException("Middlenode index out of bounds.");
                    //In case of Adaline an index always will be 0.
            return nodes[firstMiddleNode + index];
        }

        private int GetMiddleNodesCount()
        {
            return firstOutputNode - firstMiddleNode;
        }

        /// <summary>
        ///     Overridden.Returns type of the network.
        /// </summary>
        /// <returns>Returns <see cref="xpidea.neuro.net.NeuralNetworkType.nntBackProp" /> for backpropagation network.</returns>
        /// <remarks>Used for persistence purposes.</remarks>
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntBackProp;
        }

        /// <summary>
        ///     Overridden.Returns number of nodes in the input layer of the network.
        /// </summary>
        /// <returns>Nodes count.</returns>
        protected override int GetInputNodesCount()
        {
            return nodesInLayer[0];
        }

        /// <summary>
        ///     Overridden.Returns number of nodes in output layer of the network.
        /// </summary>
        /// <returns>Nodes number.</returns>
        protected override int GetOutPutNodesCount()
        {
            return nodesInLayer[layersCount - 1];
        }

        /// <summary>
        ///     Overridden.Returns output node of the network by its index.
        /// </summary>
        /// <param name="index">Output node index.</param>
        /// <returns>Output node.</returns>
        protected override NeuroNode GetOutputNode(int index)
        {
            if ((index >= OutputNodesCount) || (index < 0))
                throw new ENeuroException("OutputNode index out of bounds.");
                    //In case of Adaline an index always will be 0.
            return nodes[firstOutputNode + index];
        }

        /// <summary>
        ///     Creates a link that will be used to constract the network. In case of
        ///     <see cref="xpidea.neuro.net.backprop.BackPropagationNetwork" /> network a
        ///     <see cref="xpidea.neuro.net.backprop.BackPropagationLink" /> link is created.
        /// </summary>
        /// <returns>Link object.</returns>
        public virtual NeuroLink CreateLink()
        {
            return new BackPropagationLink();
        }

        /// <summary>
        ///     Overridden.Method that constructs network topology.
        /// </summary>
        protected override void CreateNetwork()
        {
            nodes = new NeuroNode[NodesCount];
            links = new NeuroLink[LinksCount];
            var cnt = 0;
            for (var i = 0; i < InputNodesCount; i++)
            {
                nodes[cnt] = new InputNode();
                cnt++;
            }

            firstMiddleNode = cnt;

            for (var i = 1; i < (layersCount - 1); i++)
                for (var j = 0; j < nodesInLayer[i]; j++)
                {
                    nodes[cnt] = new BackPropagationMiddleNode(LearningRate, momentum);
                    cnt++;
                }

            firstOutputNode = cnt;
            for (var i = 0; i < OutputNodesCount; i++)
            {
                nodes[cnt] = new BackPropagationOutputNode(LearningRate, momentum);
                cnt++;
            }

            for (var i = 0; i < LinksCount; i++)
                links[i] = CreateLink();
            cnt = 0;
            var l1 = 0;
            var l2 = firstMiddleNode;
            for (var i = 0; i < (layersCount - 1); i++)
            {
                for (var j = 0; j < nodesInLayer[i + 1]; j++)
                    for (var k = 0; k < nodesInLayer[i]; k++)
                    {
                        nodes[l1 + k].LinkTo(nodes[l2 + j], links[cnt]);
                        cnt++;
                    }
                l1 = l2;
                l2 += nodesInLayer[i + 1];
            }
        }

        /// <summary>
        ///     Overridden.Loads network data from a binary stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            layersCount = binaryReader.ReadInt32();
            nodesInLayer = new int[layersCount];
            for (var i = 0; i < layersCount; i++) nodesInLayer[i] = binaryReader.ReadInt32();
            base.Load(binaryReader);
        }

        /// <summary>
        ///     Overridden.Saves the network to a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            binaryWriter.Write(layersCount);
            for (var i = 0; i < layersCount; i++)
            {
                binaryWriter.Write(nodesInLayer[i]);
            }
            base.Save(binaryWriter);
        }

        /// <summary>
        ///     Overridden.Tells the network to produce output values based on its input.
        /// </summary>
        public override void Run()
        {
            LoadInputs();
            for (var i = firstMiddleNode; i < NodesCount; i++)
                nodes[i].Run();
        }

        /// <summary>
        ///     Overridden.Makes the network to learn the pattern that was just exposed to the network.
        ///     Usually executes right after <see cref="xpidea.neuro.net.backprop.BackPropagationNetwork.Run" /> method.
        ///     <seealso cref="xpidea.neuro.net.backprop.BackPropagationNetwork.Run" />
        /// </summary>
        public override void Learn()
        {
            for (var i = 0; i < OutLinks.Count; i++)
                OutputNode(i).Error = OutLinks[i].OutNode.Error;
            //TODO Fix this bug for delphi too
            for (var i = NodesCount - 1; i >= firstMiddleNode; i--)
                nodes[i].Learn();
        }

        /// <summary>
        ///     Overridden.Performs network training. Here you write the code to train your network.
        /// </summary>
        /// <param name="patterns">Set of the patterns that will be exposed to a network during the training.</param>
        public override void Train(PatternsCollection patterns)
        {
            //This method implementation is for reference only -
            //You may want to implement your own method by overriding this one.

            if (patterns != null)
            {
                var good = 0;
                double tolerance = 0.2F;
                while (good < patterns.Count) // Train until all patterns are correct
                {
                    good = 0;
                    for (var i = 0; i < patterns.Count; i++)
                    {
                        for (var k = 0; k < NodesInLayer(0); k++)
                            nodes[k].Value = patterns[i].Input[k];
                        Run();
                        for (var k = 0; k < OutputNodesCount; k++)
                            OutputNode(k).Error = patterns[i].Output[k];
                        Learn();
                        var InRange = true;
                        for (var k = 0; k < OutputNodesCount; k++)
                        {
                            if (Math.Abs(OutputNode(k).Value - patterns[i].Output[k]) >= tolerance) InRange = false;
                            //	Console.Out.WriteLine(this.OutputNode(k).Value.ToString()+"   " +this.OutputNode(k).Error.ToString());
                            // InRange = Math.Round(nodes[k].Value) == Math.Round((patterns[i]).Output[k - firstOutputNode]);
                        }
                        if (InRange)
                            good++;
                    }
                }
            }
        }

        /// <summary>
        ///     Returns number of nodes in specific layer.
        /// </summary>
        /// <param name="index">Layer index.</param>
        /// <returns></returns>
        public int NodesInLayer(int index)
        {
            return GetNodesInLayer(index);
        }
    }

    /// <summary>
    ///     A link in <see cref="xpidea.neuro.net.backprop.EpochBackPropagationNetwork" />. Implements functionality related to
    ///     an epoch network training model.
    /// </summary>
    public class EpochBackPropagationLink : BackPropagationLink
    {
        /// <summary>
        ///     Stores epoch of the link.
        /// </summary>
        protected double linkEpoch;

        /// <summary>
        ///     Property stores accumulated weight change during current epoch.
        /// </summary>
        public double LinkEpoch
        {
            get { return GetLinkEpoch(); }
            set { SetLinkEpoch(value); }
        }

        /// <summary>
        ///     Getter for <see cref="xpidea.neuro.net.backprop.EpochBackPropagationLink.LinkEpoch" /> property.
        /// </summary>
        /// <returns>Link epoch.</returns>
        protected virtual double GetLinkEpoch()
        {
            return linkEpoch;
        }

        /// <summary>
        ///     Setter for <see cref="xpidea.neuro.net.backprop.EpochBackPropagationLink.LinkEpoch" /> property.
        /// </summary>
        /// <param name="epoch">New epoch value.</param>
        protected void SetLinkEpoch(double epoch)
        {
            linkEpoch = epoch;
        }

        /// <summary>
        ///     Overridden.Loads the link data from the stream.
        /// </summary>
        /// <param name="binaryReader">Binary stream reader.</param>
        public override void Load(BinaryReader binaryReader)
        {
            base.Load(binaryReader);
            linkEpoch = binaryReader.ReadDouble();
        }

        /// <summary>
        ///     Overridden.Stores link data in a binary stream.
        /// </summary>
        /// <param name="binaryWriter">Binary stream writer.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            base.Save(binaryWriter);
            binaryWriter.Write(linkEpoch);
        }

        /// <summary>
        ///     Overridden.Updates the link weightaccording to the training model.
        ///     In this particular case accumulates deltaWeight values in
        ///     <see cref="xpidea.neuro.net.backprop.EpochBackPropagationLink.LinkEpoch" /> property.
        /// </summary>
        /// <param name="deltaWeight">Delta of the weight change.</param>
        public override void UpdateWeight(double deltaWeight)
        {
            LinkEpoch = LinkEpoch + deltaWeight; //deltaWij  (2.17)
        }

        /// <summary>
        ///     Overridden.Tells the link, that all patterns have been exposed and it's time to update link weight.
        /// </summary>
        /// <param name="epoch">Number of patterns that was exposed to the network.</param>
        public override void Epoch(int epoch)
        {
            var momentum = ((BackPropagationOutputNode) OutNode).Momentum;
            var delta = LinkEpoch/epoch;
            Weight = Weight + delta + (momentum*LinkDelta); //Wji(n+1)=Wij(n)+deltaWij(n)+alpha*deltaWij(n-1);
            LinkDelta = delta;
            LinkEpoch = 0F;
        }
    }

    /// <summary>
    ///     Nwtwork that implements Epoch training strategy.
    /// </summary>
    public class EpochBackPropagationNetwork : BackPropagationNetwork
    {
        /// <summary>
        ///     Constructs EpochBackPropagationNetwork network.
        /// </summary>
        /// <param name="learningRate">Network's leraning rate.</param>
        /// <param name="momentum">Nodes momentum.</param>
        /// <param name="nodesInEachLayer">Nodes in each layer.</param>
        public EpochBackPropagationNetwork(double learningRate, double momentum, int[] nodesInEachLayer)
            : base(learningRate, momentum, nodesInEachLayer)
        {
        }

        /// <summary>
        ///     Creates the network from a file.
        /// </summary>
        public EpochBackPropagationNetwork(string fileName) : base(fileName)
        {
        }

        /// <summary>
        ///     Overridden.Creates new <see cref="xpidea.neuro.net.backprop.EpochBackPropagationLink" />
        /// </summary>
        /// <returns></returns>
        public override NeuroLink CreateLink()
        {
            return new EpochBackPropagationLink();
        }

        /// <summary>
        ///     Overridden.Returns <see cref="xpidea.neuro.net.NeuralNetworkType.nntEpochBackProp" /> for this network.
        /// </summary>
        /// <returns></returns>
        protected override NeuralNetworkType GetNetworkType()
        {
            return NeuralNetworkType.nntEpochBackProp;
        }

        /// <summary>
        ///     Overridden.Trains the network (makes the network learn the patterns).
        /// </summary>
        /// <param name="patterns">Training patterns.</param>
        public override void Train(PatternsCollection patterns)
        {
            //This method implementation is for reference only -
            //You may want to implement your own method by overriding this one.

            if (patterns != null)
            {
                var good = 0;
                var tolerance = 0.2;
                while (good < patterns.Count) // Train until all patterns are correct
                {
                    good = 0;
                    for (var i = 0; i < patterns.Count; i++)
                    {
                        for (var k = 0; k < NodesInLayer(0); k++) nodes[k].Value = (patterns[i]).Input[k];
                        for (var j = firstMiddleNode; j < NodesCount; j++) nodes[j].Run();
                        for (var k = firstOutputNode; k < NodesCount; k++)
                            nodes[k].Error = (patterns[i]).Output[k - firstOutputNode];
                        for (var j = NodesCount - 1; j >= firstMiddleNode; j--)
                            nodes[j].Learn();
                        var InRange = true;
                        for (var k = 0; k < OutputNodesCount; k++)
                        {
                            if (Math.Abs(OutputNode(k).Value - (patterns[i]).Output[k]) >= tolerance) InRange = false;
                            //Console.Out.WriteLine(this.OutputNode(k).Value.ToString()+"   " +this.OutputNode(k).Error.ToString());
                        }
                        if (InRange) good++;
                    }
                    Epoch(patterns.Count);
                }
            }
        }

        /// <summary>
        ///     Overridden.Finalizes trainig cycle of the network. Used by <see cref="xpidea.neuro.net.NeuralNetwork.Train" />
        ///     method of the network.
        /// </summary>
        /// <param name="epoch">Number of patterns was exposed to the network.</param>
        public override void Epoch(int epoch)
        {
            foreach (var link in links) link.Epoch(epoch);
        }
    }

    /// <summary>
    ///     A link for <see cref="xpidea.neuro.net.backprop.EpochBackPropagationNetwork" /> network.
    /// </summary>
    public class BackPropagationRPROPLink : EpochBackPropagationLink
    {
        private double decayTerm; //Decay
        private double eta;
        private double etaMinus; //learning decrease
        private double etaPlus; //learning increase
        private double initEta; //Initial Eta;
        private double maxDelta; //maximum learning
        private double minDelta; //minimum learning

        /// <summary>
        ///     Overridden.Initializes private variables.
        /// </summary>
        protected override void DoAfterCreate()
        {
            base.DoAfterCreate();
            etaPlus = 1.2F; //learning increase
            etaMinus = 0.5F; //learning decrease
            initEta = 0.05F; //Initial Eta;
            maxDelta = 50.0F; //maximum learning
            minDelta = 1.0E-6F; //minimum learning
            decayTerm = 0.0F; //Decay
            eta = initEta;
        }

        /// <summary>
        ///     Overridden.Tells the link, that all patterns have been exposed and it's time to update link weight.
        /// </summary>
        /// <param name="epoch">Number of patterns that was exposed to the network.</param>
        /// <remarks>Implements RPROP training algorithm.</remarks>
        public override void Epoch(int epoch)
        {
            var delta = -LinkEpoch/epoch;
            var deltaOld = LinkDelta;
            var deltaProduct = deltaOld*delta;
            var deltaSign = Math.Sign(delta);
            delta = delta - decayTerm*Weight;
            if (deltaProduct >= 0)
            {
                if (deltaProduct > 0) eta = Math.Min(eta*etaPlus, maxDelta);
                Weight = Weight - deltaSign*eta;
                deltaOld = delta;
            }
            else if (deltaProduct < 0)
            {
                eta = Math.Max(eta*etaMinus, minDelta);
                deltaOld = 0.0;
            }
            LinkDelta = deltaOld;
            LinkEpoch = 0;
        }
    }

    /// <summary>
    ///     A Backpropagation network with epoch training implementing RPOP training algorithm.
    /// </summary>
    public class BackPropagationRPROPNetwork : EpochBackPropagationNetwork
    {
        /// <summary>
        ///     Creates the network from a file.
        /// </summary>
        public BackPropagationRPROPNetwork(string fileName) : base(fileName)
        {
        }

        /// <summary>
        ///     Creates the network.
        /// </summary>
        /// <param name="nodesInEachLayer">Nodes in each layer of the network.</param>
        public BackPropagationRPROPNetwork(int[] nodesInEachLayer) : base(1, 0, nodesInEachLayer)
        {
        }

        /// <summary>
        ///     Overridden.Creates new <see cref="xpidea.neuro.net.backprop.EpochBackPropagationLink" />
        /// </summary>
        /// <returns>Link.</returns>
        public override NeuroLink CreateLink()
        {
            return new BackPropagationRPROPLink();
        }
    }
}