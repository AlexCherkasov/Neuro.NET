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
using System.Collections;
using System.IO;

namespace xpidea.neuro.net.patterns
{
    /// <summary>
    ///     A class representing single training pattern and is used to train a neural network. Contains input data and
    ///     expected results arrays.
    /// </summary>
    public class Pattern : NeuroObject
    {
        /// <summary>
        ///     Constructor. Creates new instance of the pattern.
        /// </summary>
        /// <param name="inputsCount">Number of input values in a pattern.</param>
        /// <param name="outputsCount">Number of output/result values in the pattern.</param>
        public Pattern(int inputsCount, int outputsCount)
        {
            InputsCount = inputsCount;
            OutputsCount = outputsCount;
            Input = new double[InputsCount];
            Output = new double[OutputsCount];
        }

        /// <summary>
        ///     Array of input parameters. Each value usually in a range from -1..1
        /// </summary>
        public double[] Input { get; private set; }

        /// <summary>
        ///     Array of output parameters. Each value usually in a range from -1..1
        /// </summary>
        public double[] Output { get; private set; }

        /// <summary>
        ///     Number of input parameters.
        /// </summary>
        public int InputsCount { get; private set; }

        /// <summary>
        ///     Number of output parameters.
        /// </summary>
        public int OutputsCount { get; private set; }

        /// <summary>
        ///     Overridden.Saves the pattern to a stream.
        /// </summary>
        /// <param name="binaryWriter">A BinaryWriter used to write a stream.</param>
        public override void Save(BinaryWriter binaryWriter)
        {
            binaryWriter.Write(InputsCount);
            binaryWriter.Write(OutputsCount);
            foreach (var d in Input) binaryWriter.Write(d);
            foreach (var d in Output) binaryWriter.Write(d);
        }

        /// <summary>
        ///     Overridden.Loads the pattern from a stream.
        /// </summary>
        /// <param name="binaryReader">A BinaryReader used to read the stream.</param>
        public override void Load(BinaryReader binaryReader)
        {
            InputsCount = binaryReader.ReadInt32();
            OutputsCount = binaryReader.ReadInt32();
            Input = new double[InputsCount];
            Output = new double[OutputsCount];
            for (var i = 0; i < InputsCount; i++) Input[i] = binaryReader.ReadDouble();
            for (var i = 0; i < OutputsCount; i++) Output[i] = binaryReader.ReadDouble();
        }
    }


    /// <summary>
    ///     Represents vollection of <see cref="xpidea.neuro.net.NeuroObject" />
    /// </summary>
    public abstract class NeuroObjectCollection : CollectionBase
    {
        /// <summary>
        ///     Creates the collection.
        /// </summary>
        internal NeuroObjectCollection()
        {
        }

        /// <summary>
        ///     Constructor. Creates the collection and loads content from a file.
        /// </summary>
        /// <param name="fileName">Name of the file to load data from.</param>
        internal NeuroObjectCollection(string fileName) : this()
        {
            LoadFromFile(fileName);
        }

        /// <summary>
        /// </summary>
        /// <param name="binaryWriter"></param>
        public virtual void Save(BinaryWriter binaryWriter)
        {
            binaryWriter.Write(Count);
            foreach (NeuroObject obj in this) obj.Save(binaryWriter);
        }

        /// <summary>
        /// </summary>
        /// <param name="binaryReader"></param>
        public virtual void Load(BinaryReader binaryReader)
        {
            var itemsCount = binaryReader.ReadInt32();
            for (var i = 0; i < itemsCount; i++)
            {
                var no = CreateContainigObject();
                no.Load(binaryReader);
                List.Add(no);
            }
        }

        /// <summary>
        ///     Constucts the object that could be stored in this collection.
        /// </summary>
        /// <returns>Object.</returns>
        protected abstract NeuroObject CreateContainigObject();

        /// <summary>
        ///     Loads collection from a file.
        /// </summary>
        /// <param name="fileName"></param>
        public void LoadFromFile(string fileName)
        {
            Stream stream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
            var binaryReader = new BinaryReader(stream);
            Load(binaryReader);
            stream.Close();
        }

        /// <summary>
        ///     Stores collection of the objectects into a file in a binary format.
        /// </summary>
        /// <param name="fileName">File name.</param>
        public virtual void SaveToFile(string fileName)
        {
            Stream stream = new FileStream(fileName, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
            var binaryWriter = new BinaryWriter(stream);
            Save(binaryWriter);
            stream.Close();
        }
    }

    /// <summary>
    ///     A class representing collection of the patterns.
    ///     <para>
    ///         A collection that stores <see cref='xpidea.neuro.net.patterns.Pattern' /> objects.
    ///     </para>
    /// </summary>
    /// <seealso cref='xpidea.neuro.net.patterns.PatternsCollection' />
    [Serializable]
    public class PatternsCollection : NeuroObjectCollection
    {
        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref="xpidea.neuro.net.patterns.PatternsCollection" />.
        ///     </para>
        /// </summary>
        public PatternsCollection()
        {
        }

        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> based on another
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" />.
        ///     </para>
        /// </summary>
        /// <param name='value'>
        ///     A <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> from which the contents are copied
        /// </param>
        public PatternsCollection(PatternsCollection value)
        {
            AddRange(value);
        }

        /// <summary>
        ///     <para>
        ///         Initializes a new instance of <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> containing any array
        ///         of <see cref='xpidea.neuro.net.patterns.Pattern' /> objects.
        ///     </para>
        /// </summary>
        /// <param name='value'>
        ///     A array of <see cref='xpidea.neuro.net.patterns.Pattern' /> objects with which to intialize the collection
        /// </param>
        public PatternsCollection(Pattern[] value)
        {
            AddRange(value);
        }

        /// <summary>
        ///     Constructor. Creates new cllection of the patterns.
        /// </summary>
        /// <param name="patternsCount">Numeber of patterns in this collection.</param>
        /// <param name="inputsCount">Number of inputs in each pattern.</param>
        /// <param name="outputsCount">Number of outputs in each pattern.</param>
        public PatternsCollection(int patternsCount, int inputsCount, int outputsCount)
        {
            for (var i = 0; i < patternsCount; i++)
                Add(new Pattern(inputsCount, outputsCount));
        }

        /// <summary>
        ///     Constructor. Creates the collection and loads content from a file.
        /// </summary>
        /// <param name="fileName">Name of the file to load data from.</param>
        public PatternsCollection(string fileName) : base(fileName)
        {
        }

        /// <summary>
        ///     <para>Represents the entry at the specified index of the <see cref='xpidea.neuro.net.patterns.Pattern' />.</para>
        /// </summary>
        /// <param name='index'>
        ///     <para>The zero-based index of the entry to locate in the collection.</para>
        /// </param>
        /// <value>
        ///     <para> The entry at the specified index of the collection.</para>
        /// </value>
        /// <exception cref='System.ArgumentOutOfRangeException'>
        ///     <paramref name='index' /> is outside the valid range of indexes
        ///     for the collection.
        /// </exception>
        public Pattern this[int index]
        {
            get { return ((Pattern)(List[index])); }
            set { List[index] = value; }
        }

        /// <summary>
        ///     Overridden.Creates new <see cref="xpidea.neuro.net.patterns.Pattern" /> object.
        /// </summary>
        /// <returns></returns>
        protected override NeuroObject CreateContainigObject()
        {
            return new Pattern(0, 0);
        }

        /// <summary>
        ///     <para>
        ///         Adds a <see cref='xpidea.neuro.net.patterns.Pattern' /> with the specified value to the
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.patterns.Pattern' /> to add.</param>
        /// <returns>
        ///     <para>The index at which the new element was inserted.</para>
        /// </returns>
        /// <seealso cref="xpidea.neuro.net.patterns.PatternsCollection.AddRange" />
        public int Add(Pattern value)
        {
            return List.Add(value);
        }

        /// <summary>
        ///     <para>
        ///         Copies the elements of an array to the end of the <see cref="xpidea.neuro.net.patterns.PatternsCollection" />
        ///         .
        ///     </para>
        /// </summary>
        /// <param name="value">
        ///     An array of type <see cref="xpidea.neuro.net.patterns.Pattern" /> containing the objects to add to the collection.
        /// </param>
        /// <seealso cref="xpidea.neuro.net.patterns.PatternsCollection.Add" />
        public void AddRange(Pattern[] value)
        {
            for (var i = 0; (i < value.Length); i = (i + 1))
            {
                Add(value[i]);
            }
        }

        /// <summary>
        ///     <para>
        ///         Adds the contents of another <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> to the end of the
        ///         collection.
        ///     </para>
        /// </summary>
        /// <param name="value">
        ///     A <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> containing the objects to add to the collection.
        /// </param>
        /// <seealso cref="xpidea.neuro.net.patterns.PatternsCollection.Add" />
        public void AddRange(PatternsCollection value)
        {
            for (var i = 0; (i < value.Count); i = (i + 1))
            {
                Add(value[i]);
            }
        }

        /// <summary>
        ///     <para>
        ///         Gets a value indicating whether the
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> contains the specified
        ///         <see cref='xpidea.neuro.net.patterns.Pattern' />.
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.patterns.Pattern' /> to locate.</param>
        /// <returns>
        ///     <para>
        ///         <see langword='true' /> if the <see cref='xpidea.neuro.net.patterns.Pattern' /> is contained in the collection;
        ///         otherwise, <see langword='false' />.
        ///     </para>
        /// </returns>
        /// <seealso cref='xpidea.neuro.net.patterns.PatternsCollection.IndexOf' />
        public bool Contains(Pattern value)
        {
            return List.Contains(value);
        }

        /// <summary>
        ///     <para>
        ///         Copies the <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> values to a one-dimensional
        ///         <see cref="System.Array" /> instance at the
        ///         specified index.
        ///     </para>
        /// </summary>
        /// <param name="array">
        ///     <para>
        ///         The one-dimensional <see cref="System.Array" /> that is the destination of the values copied from
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        ///     </para>
        /// </param>
        /// <param name="index">The index in <paramref name="array" /> where copying begins.</param>
        /// <exception cref="System.ArgumentException">
        ///     <para><paramref name="array" /> is multidimensional.</para>
        ///     <para>-or-</para>
        ///     <para>
        ///         The number of elements in the <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> is greater than the
        ///         available space between <paramref name="index" /> and the end of <paramref name="array" />.
        ///     </para>
        /// </exception>
        /// <exception cref="System.ArgumentNullException"><paramref name="array" /> is <see langword="null" />. </exception>
        /// <exception cref="System.ArgumentOutOfRangeException">
        ///     <paramref name="arrayIndex" /> is less than
        ///     <paramref name="array" />'s lowbound.
        /// </exception>
        /// <seealso cref="System.Array" />
        public void CopyTo(Pattern[] array, int index)
        {
            List.CopyTo(array, index);
        }

        /// <summary>
        ///     <para>
        ///         Returns the index of a <see cref='xpidea.neuro.net.patterns.Pattern' /> in
        ///         the <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        ///     </para>
        /// </summary>
        /// <param name='value'>The <see cref='xpidea.neuro.net.patterns.Pattern' /> to locate.</param>
        /// <returns>
        ///     <para>
        ///         The index of the <see cref='xpidea.neuro.net.patterns.Pattern' /> of <paramref name='value' /> in the
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" />, if found; otherwise, -1.
        ///     </para>
        /// </returns>
        /// <seealso cref='xpidea.neuro.net.patterns.PatternsCollection.Contains' />
        public int IndexOf(Pattern value)
        {
            return List.IndexOf(value);
        }

        /// <summary>
        ///     <para>
        ///         Inserts a <see cref="xpidea.neuro.net.patterns.Pattern" /> into the
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> at the specified index.
        ///     </para>
        /// </summary>
        /// <param name="index">The zero-based index where <paramref name="value" /> should be inserted.</param>
        /// <param name=" value">The <see cref="xpidea.neuro.net.patterns.Pattern" /> to insert.</param>
        /// <seealso cref="xpidea.neuro.net.patterns.PatternsCollection.Add" />
        public void Insert(int index, Pattern value)
        {
            List.Insert(index, value);
        }

        /// <summary>
        ///     <para>
        ///         Returns an enumerator that can iterate through
        ///         the <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        ///     </para>
        /// </summary>
        /// <returns>
        ///     <para>None.</para>
        /// </returns>
        /// <seealso cref='System.Collections.IEnumerator' />
        public new CustomPatternEnumerator GetEnumerator()
        {
            return new CustomPatternEnumerator(this);
        }

        /// <summary>
        ///     <para>
        ///         Removes a specific <see cref="xpidea.neuro.net.patterns.Pattern" /> from the
        ///         <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        ///     </para>
        /// </summary>
        /// <param name="value">
        ///     The <see cref="xpidea.neuro.net.patterns.Pattern" /> to remove from the
        ///     <see cref="xpidea.neuro.net.patterns.PatternsCollection" /> .
        /// </param>
        /// <exception cref="System.ArgumentException"><paramref name="value" /> is not found in the Collection. </exception>
        public void Remove(Pattern value)
        {
            List.Remove(value);
        }

        /// <summary>
        /// </summary>
        public class CustomPatternEnumerator : object, IEnumerator
        {
            private readonly IEnumerator baseEnumerator;
            private readonly IEnumerable temp;

            /// <summary>
            /// </summary>
            /// <param name="mappings"></param>
            public CustomPatternEnumerator(PatternsCollection mappings)
            {
                temp = mappings;
                baseEnumerator = temp.GetEnumerator();
            }

            /// <summary>
            /// </summary>
            public Pattern Current
            {
                get { return ((Pattern)(baseEnumerator.Current)); }
            }

            object IEnumerator.Current
            {
                get { return baseEnumerator.Current; }
            }

            bool IEnumerator.MoveNext()
            {
                return baseEnumerator.MoveNext();
            }

            void IEnumerator.Reset()
            {
                baseEnumerator.Reset();
            }

            /// <summary>
            /// </summary>
            /// <returns></returns>
            public bool MoveNext()
            {
                return baseEnumerator.MoveNext();
            }

            /// <summary>
            /// </summary>
            public void Reset()
            {
                baseEnumerator.Reset();
            }
        }
    }
}