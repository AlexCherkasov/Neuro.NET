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
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using xpidea.neuro.net.backprop;
using xpidea.neuro.net.patterns;

namespace xpidea.neuro.net.examples.backprop.ocr
{
    /// <summary>
    ///     Summary description for Form1.
    /// </summary>
    public class Form1 : Form
    {
        public static bool IsTerminated;
        public static int aMatrixDim = 10;
        public static int aFirstChar = 'A';
        public static int aLastChar = 'z';
        public static int aCharsCount = aLastChar - aFirstChar + 1;
        public OCRNetwork backpropNetwork;
        public PatternsCollection trainingPatterns;

        public PatternsCollection CreateTrainingPatterns(Font font)
        {
            var result = new PatternsCollection(aCharsCount, aMatrixDim*aMatrixDim, 1);
            for (var i = 0; i < aCharsCount; i++)
            {
                var aBitMatrix = CharToDoubleArray(Convert.ToChar(aFirstChar + i), font, aMatrixDim, 0);
                for (var j = 0; j < aMatrixDim*aMatrixDim; j++)
                    result[i].Input[j] = aBitMatrix[j];
                result[i].Output[0] = i;
            }
            return result;
        }

        private void ShowNoise(Size sz, Graphics g, int noisePercent)
        {
            var range = sz.Height*sz.Width*noisePercent/200;
            for (var i = 0; i < range; i++)
            {
                var x = (int) NeuroObject.Random(0, sz.Width);
                var y = (int) NeuroObject.Random(0, sz.Height);
                var r = new Rectangle(x, y, 0, 0);
                r.Inflate(1, 1);
                Brush b;
                if ((NeuroObject.Random(0, 100)) > 80) //80% is black noise, 20% is white noise
                    b = new SolidBrush(Color.White);
                else
                    b = new SolidBrush(Color.Black);

                g.FillRectangle(b, r);
                b.Dispose();
            }
        }

        public double[] CharToDoubleArray(char aChar, Font aFont, int aArrayDim, int aAddNoisePercent)
        {
            var result = new double[aArrayDim*aArrayDim];
            var gr = label5.CreateGraphics();
            var size = Size.Round(gr.MeasureString(aChar.ToString(), aFont));
            var aSrc = new Bitmap(size.Width, size.Height);
            var bmp = Graphics.FromImage(aSrc);
            bmp.SmoothingMode = SmoothingMode.None;
            bmp.InterpolationMode = InterpolationMode.NearestNeighbor;
            bmp.Clear(Color.White);
            bmp.DrawString(aChar.ToString(), aFont, new SolidBrush(Color.Black), new Point(0, 0), new StringFormat());
            ShowNoise(size, bmp, aAddNoisePercent);
            pictureBox1.Image = aSrc;
            Application.DoEvents();
            var xStep = aSrc.Width/(double) aArrayDim;
            var yStep = aSrc.Height/(double) aArrayDim;
            for (var i = 0; i < aSrc.Width; i++)
                for (var j = 0; j < aSrc.Height; j++)
                {
                    var x = (int) ((i/xStep));
                    var y = (int) (j/yStep);
                    var c = aSrc.GetPixel(i, j);
                    result[y*x + y] += Math.Sqrt(c.R*c.R + c.B*c.B + c.G*c.G);
                        //Convert to BW, I guess I can use B component of Alpha color space too...
                }
            return Scale(result);
        }

        private double MaxOf(double[] src)
        {
            var res = double.NegativeInfinity;
            foreach (var d in src)
                if (d > res) res = d;
            return res;
        }

        private double[] Scale(double[] src)
        {
            var max = MaxOf(src);
            if (max != 0)
            {
                for (var i = 0; i < src.Length; i++)
                    src[i] = src[i]/max;
            }
            return src;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            aFirstChar = textBox1.Text[0];
            aLastChar = textBox2.Text[0];
            aCharsCount = aLastChar - aFirstChar + 1;
            label5.Text = "";
            var chCnt = aCharsCount;
            if (aCharsCount > 50)
                chCnt = 50;
            for (var i = 0; i < chCnt; i++)
                label5.Text += Convert.ToChar(aFirstChar + i) + " ";
            trainingPatterns = CreateTrainingPatterns(label5.Font);
            tabControl1.SelectedTab = tabPage2;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            backpropNetwork = new OCRNetwork(this,
                new int[3] {aMatrixDim*aMatrixDim, (aMatrixDim*aMatrixDim + aCharsCount)/2, aCharsCount});
            tabControl1.SelectedTab = tabPage3;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (backpropNetwork == null)
            {
                MessageBox.Show("Please create the network first!");
                return;
            }
            if (trainingPatterns == null)
            {
                MessageBox.Show("Please create training patterns on STEP 1");
                return;
            }
            progressBar1.Maximum = aCharsCount;
            label9.Text =
                "While the network is training you can proceed to the STEP 4 and observe how the recognition quality progress during the training.";
            backpropNetwork.Train(trainingPatterns);
            MessageBox.Show("Network training successfully complete!");
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (backpropNetwork == null)
            {
                MessageBox.Show("Network is not yet created.");
                return;
            }
            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
                backpropNetwork.SaveToFile(saveFileDialog1.FileName);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (backpropNetwork == null)
            {
                MessageBox.Show("Network is not yet created.");
                return;
            }
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
                backpropNetwork.LoadFromFile(openFileDialog1.FileName);
        }

        private void Form1_Closing(object sender, CancelEventArgs e)
        {
            IsTerminated = true;
        }

        private void trackBar3_Scroll(object sender, EventArgs e)
        {
            label8.Text = "Add noise to the patterns  ( " + trackBar3.Value + "% )";
        }

        private void trackBar4_Scroll(object sender, EventArgs e)
        {
            label10.Text = "Noise level ( " + trackBar4.Value + "% )";
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (backpropNetwork == null)
            {
                MessageBox.Show("Please go to STEP 2 and create the network.");
                return;
            }
            label14.Text = textBox3.Text;
            if (textBox3.Text.Length > 0)
            {
                var aInput = CharToDoubleArray(textBox3.Text[0], label5.Font, aMatrixDim, trackBar4.Value);
                for (var i = 0; i < backpropNetwork.InputNodesCount; i++)
                    backpropNetwork.InputNode(i).Value = aInput[i];
                backpropNetwork.Run();
                label15.Text = Convert.ToChar(aFirstChar + backpropNetwork.BestNodeIndex).ToString();
            }
        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            aMatrixDim = (int) numericUpDown1.Value;
        }

        public class OCRNetwork : BackPropagationRPROPNetwork
        {
            private readonly Form1 owner;

            public OCRNetwork(Form1 owner, int[] nodesInEachLayer) : base(nodesInEachLayer)
            {
                this.owner = owner;
            }

            public int BestNodeIndex
            {
                get
                {
                    var result = -1;
                    double aMaxNodeValue = 0;
                    var aMinError = double.PositiveInfinity;
                    for (var i = 0; i < OutputNodesCount; i++)
                    {
                        var node = OutputNode(i);
                        if ((node.Value > aMaxNodeValue) || ((node.Value >= aMaxNodeValue) && (node.Error < aMinError)))
                        {
                            aMaxNodeValue = node.Value;
                            aMinError = node.Error;
                            result = i;
                        }
                    }
                    return result;
                }
            }

            public void AddNoiseToInputPattern(int levelPercent)
            {
                var i = ((NodesInLayer(0) - 1)*levelPercent)/100;
                while (i > 0)
                {
                    nodes[(int) (Random(0, NodesInLayer(0) - 1))].Value = Random(0, 100);
                    i--;
                }
            }

            public override void Train(PatternsCollection patterns)
            {
                var iteration = 0;
                if (patterns != null)
                {
                    double error = 0;
                    var good = 0;
                    while (good < patterns.Count) // Train until all patterns are correct
                    {
                        if (IsTerminated) return;
                        error = 0;
                        owner.progressBar1.Value = good;
                        owner.label16.Text = "Training progress: " + ((good*100)/owner.progressBar1.Maximum) + "%";
                        good = 0;
                        for (var i = 0; i < patterns.Count; i++)
                        {
                            for (var k = 0; k < NodesInLayer(0); k++)
                                nodes[k].Value = patterns[i].Input[k];
                            AddNoiseToInputPattern(owner.trackBar3.Value);
                            Run();
                            var idx = (int) patterns[i].Output[0];
                            for (var k = 0; k < OutputNodesCount; k++)
                            {
                                error += Math.Abs(OutputNode(k).Error);
                                if (k == idx)
                                    OutputNode(k).Error = 1;
                                else
                                    OutputNode(k).Error = 0;
                            }
                            Learn();
                            if (BestNodeIndex == idx)
                                good++;

                            iteration ++;
                            Application.DoEvents();
                        }

                        foreach (var link in links) ((EpochBackPropagationLink) link).Epoch(patterns.Count);

                        if ((iteration%2) == 0)
                            owner.label17.Text = "AVG Error: " + (error/OutputNodesCount) + "  Iteration: " + iteration;
                    }
                    owner.label17.Text = "AVG Error: " + (error/OutputNodesCount) + "  Iteration: " + iteration;
                }
            }
        }

        #region Variables

        private PictureBox pictureBox1;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private TabPage tabPage3;
        private TabPage tabPage4;
        private Label label1;
        private Label label2;
        private Label label3;
        private Label label4;
        private Label label5;
        private Button button1;
        private Button button2;
        private Button button3;
        private Button button4;
        private Button button5;
        private TrackBar trackBar3;
        private Label label8;
        private Label label9;
        private Label label10;
        private TrackBar trackBar4;
        private Label label12;
        private Label label13;
        private Label label14;
        private Label label15;
        private ProgressBar progressBar1;
        private Label label16;
        private Label label17;
        private OpenFileDialog openFileDialog1;
        private SaveFileDialog saveFileDialog1;
        private TextBox textBox1;
        private TextBox textBox2;
        private Label label6;
        private TextBox textBox3;
        private Button button6;
        private Label label7;
        private NumericUpDown numericUpDown1;

        /// <summary>
        ///     Required designer variable.
        /// </summary>
        private readonly Container components = null;

        #endregion

        #region OtherJunk

        public Form1()
        {
            //
            // Required for Windows Form Designer support
            //
            InitializeComponent();

            //
            // TODO: Add any constructor code after InitializeComponent call
            //
            label5.Text = "";
            for (var i = 0; i < aCharsCount; i++)
                label5.Text += Convert.ToChar(aFirstChar + i) + " ";
        }

        /// <summary>
        ///     Clean up any resources being used.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (components != null)
                {
                    components.Dispose();
                }
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///     Required method for Designer support - do not modify
        ///     the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.button1 = new System.Windows.Forms.Button();
            this.label5 = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.button2 = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.label17 = new System.Windows.Forms.Label();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.trackBar3 = new System.Windows.Forms.TrackBar();
            this.button5 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.button6 = new System.Windows.Forms.Button();
            this.textBox3 = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.label10 = new System.Windows.Forms.Label();
            this.trackBar4 = new System.Windows.Forms.TrackBar();
            this.label4 = new System.Windows.Forms.Label();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.label7 = new System.Windows.Forms.Label();
            this.numericUpDown1 = new System.Windows.Forms.NumericUpDown();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize) (this.trackBar3)).BeginInit();
            this.tabPage4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize) (this.trackBar4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize) (this.numericUpDown1)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.Dock = System.Windows.Forms.DockStyle.Right;
            this.pictureBox1.Location = new System.Drawing.Point(392, 0);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(200, 266);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.tabPage1,
                this.tabPage2,
                this.tabPage3,
                this.tabPage4
            });
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(392, 266);
            this.tabControl1.TabIndex = 1;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.textBox2,
                this.textBox1,
                this.button1,
                this.label5,
                this.label1,
                this.label6
            });
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Size = new System.Drawing.Size(384, 240);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Step 1";
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(146, 43);
            this.textBox2.MaxLength = 1;
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(14, 20);
            this.textBox2.TabIndex = 4;
            this.textBox2.Text = "z";
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(90, 43);
            this.textBox1.MaxLength = 1;
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(14, 20);
            this.textBox1.TabIndex = 3;
            this.textBox1.Text = "A";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(112, 208);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(152, 24);
            this.button1.TabIndex = 2;
            this.button1.Text = "Generate training patterns";
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label5
            // 
            this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label5.Location = new System.Drawing.Point(16, 88);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(352, 112);
            this.label5.TabIndex = 1;
            this.label5.Text = "label5";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label1
            // 
            this.label1.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.label1.Dock = System.Windows.Forms.DockStyle.Top;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label1.ForeColor = System.Drawing.SystemColors.Window;
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(384, 32);
            this.label1.TabIndex = 0;
            this.label1.Text = "   Step1:   Generate neural network training patterns";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(40, 48);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(100, 16);
            this.label6.TabIndex = 5;
            this.label6.Text = "From                  to                ";
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.numericUpDown1,
                this.label7,
                this.button2,
                this.label2
            });
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Size = new System.Drawing.Size(384, 240);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Step 2";
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(88, 160);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(200, 24);
            this.button2.TabIndex = 6;
            this.button2.Text = "Create the network";
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // label2
            // 
            this.label2.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.label2.Dock = System.Windows.Forms.DockStyle.Top;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label2.ForeColor = System.Drawing.SystemColors.Window;
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(384, 32);
            this.label2.TabIndex = 1;
            this.label2.Text = "   Step2:  Create Backpropagation Neural Network";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.label17,
                this.progressBar1,
                this.label9,
                this.label8,
                this.trackBar3,
                this.button5,
                this.button4,
                this.button3,
                this.label3
            });
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(384, 240);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Step 3";
            // 
            // label17
            // 
            this.label17.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.label17.Location = new System.Drawing.Point(0, 200);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(384, 16);
            this.label17.TabIndex = 9;
            this.label17.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // progressBar1
            // 
            this.progressBar1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.progressBar1.Location = new System.Drawing.Point(0, 216);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(384, 24);
            this.progressBar1.TabIndex = 8;
            // 
            // label9
            // 
            this.label9.Location = new System.Drawing.Point(184, 120);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(176, 64);
            this.label9.TabIndex = 7;
            // 
            // label8
            // 
            this.label8.Location = new System.Drawing.Point(184, 40);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(184, 23);
            this.label8.TabIndex = 6;
            this.label8.Text = "Add noise to the patterns";
            // 
            // trackBar3
            // 
            this.trackBar3.Location = new System.Drawing.Point(184, 64);
            this.trackBar3.Maximum = 100;
            this.trackBar3.Name = "trackBar3";
            this.trackBar3.Size = new System.Drawing.Size(176, 45);
            this.trackBar3.TabIndex = 5;
            this.trackBar3.TickFrequency = 5;
            this.trackBar3.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.trackBar3.Scroll += new System.EventHandler(this.trackBar3_Scroll);
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(16, 120);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(120, 24);
            this.button5.TabIndex = 4;
            this.button5.Text = "Load trained network";
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(16, 88);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(120, 24);
            this.button4.TabIndex = 3;
            this.button4.Text = "Save trained network";
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(16, 56);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(120, 24);
            this.button3.TabIndex = 2;
            this.button3.Text = "Train the network";
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // label3
            // 
            this.label3.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.label3.Dock = System.Windows.Forms.DockStyle.Top;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label3.ForeColor = System.Drawing.SystemColors.Window;
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(384, 32);
            this.label3.TabIndex = 1;
            this.label3.Text = "   Step3:  Train the network";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // tabPage4
            // 
            this.tabPage4.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.button6,
                this.textBox3,
                this.label16,
                this.label15,
                this.label14,
                this.label13,
                this.label12,
                this.label10,
                this.trackBar4,
                this.label4
            });
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(384, 240);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Step 4";
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(176, 192);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(112, 24);
            this.button6.TabIndex = 16;
            this.button6.Text = "Enter the Character";
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // textBox3
            // 
            this.textBox3.Location = new System.Drawing.Point(136, 192);
            this.textBox3.MaxLength = 1;
            this.textBox3.Name = "textBox3";
            this.textBox3.Size = new System.Drawing.Size(24, 20);
            this.textBox3.TabIndex = 15;
            this.textBox3.Text = "";
            // 
            // label16
            // 
            this.label16.Dock = System.Windows.Forms.DockStyle.Top;
            this.label16.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label16.Location = new System.Drawing.Point(0, 32);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(384, 16);
            this.label16.TabIndex = 14;
            this.label16.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label15
            // 
            this.label15.Font = new System.Drawing.Font("Microsoft Sans Serif", 36F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label15.Location = new System.Drawing.Point(248, 80);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(80, 56);
            this.label15.TabIndex = 13;
            this.label15.Text = "A";
            // 
            // label14
            // 
            this.label14.BackColor = System.Drawing.SystemColors.Window;
            this.label14.Font = new System.Drawing.Font("Microsoft Sans Serif", 26.25F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label14.Location = new System.Drawing.Point(80, 88);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(40, 48);
            this.label14.TabIndex = 12;
            this.label14.Text = "A";
            // 
            // label13
            // 
            this.label13.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label13.Location = new System.Drawing.Point(168, 104);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(44, 16);
            this.label13.TabIndex = 11;
            this.label13.Text = "------>";
            // 
            // label12
            // 
            this.label12.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label12.Location = new System.Drawing.Point(24, 56);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(328, 16);
            this.label12.TabIndex = 10;
            this.label12.Text = "You\'ve just entered                            Recognized";
            // 
            // label10
            // 
            this.label10.Location = new System.Drawing.Point(16, 144);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(128, 16);
            this.label10.TabIndex = 8;
            this.label10.Text = "Noise level (%)";
            this.label10.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // trackBar4
            // 
            this.trackBar4.Location = new System.Drawing.Point(144, 136);
            this.trackBar4.Maximum = 100;
            this.trackBar4.Name = "trackBar4";
            this.trackBar4.Size = new System.Drawing.Size(176, 45);
            this.trackBar4.TabIndex = 7;
            this.trackBar4.TickFrequency = 5;
            this.trackBar4.TickStyle = System.Windows.Forms.TickStyle.Both;
            this.trackBar4.Scroll += new System.EventHandler(this.trackBar4_Scroll);
            // 
            // label4
            // 
            this.label4.BackColor = System.Drawing.SystemColors.InactiveCaption;
            this.label4.Dock = System.Windows.Forms.DockStyle.Top;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Bold,
                System.Drawing.GraphicsUnit.Point, ((System.Byte) (0)));
            this.label4.ForeColor = System.Drawing.SystemColors.Window;
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(384, 32);
            this.label4.TabIndex = 1;
            this.label4.Text = "   Step4:  Testing";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.DefaultExt = "neuro";
            this.openFileDialog1.FileName = "OCRNetwork";
            this.openFileDialog1.Filter = "Neural network (*.neuro)|*.neuro";
            this.openFileDialog1.Title = "Load neural network";
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.DefaultExt = "neuro";
            this.saveFileDialog1.FileName = "OCRNetwork";
            this.saveFileDialog1.Filter = "Neural network (*.neuro)|*.neuro";
            this.saveFileDialog1.Title = "Store the network";
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(16, 64);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(136, 16);
            this.label7.TabIndex = 7;
            this.label7.Text = "Network Matrix Dimention";
            // 
            // numericUpDown1
            // 
            this.numericUpDown1.Increment = new System.Decimal(new int[]
            {
                5,
                0,
                0,
                0
            });
            this.numericUpDown1.Location = new System.Drawing.Point(153, 63);
            this.numericUpDown1.Minimum = new System.Decimal(new int[]
            {
                5,
                0,
                0,
                0
            });
            this.numericUpDown1.Name = "numericUpDown1";
            this.numericUpDown1.Size = new System.Drawing.Size(48, 20);
            this.numericUpDown1.TabIndex = 8;
            this.numericUpDown1.Value = new System.Decimal(new int[]
            {
                10,
                0,
                0,
                0
            });
            this.numericUpDown1.ValueChanged += new System.EventHandler(this.numericUpDown1_ValueChanged);
            // 
            // Form1
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(592, 266);
            this.Controls.AddRange(new System.Windows.Forms.Control[]
            {
                this.tabControl1,
                this.pictureBox1
            });
            this.Name = "Form1";
            this.Text = "XPidea  -=Simple OCR Demo (Backprop network RPROP algorithm)=-";
            this.Closing += new System.ComponentModel.CancelEventHandler(this.Form1_Closing);
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage2.ResumeLayout(false);
            this.tabPage3.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize) (this.trackBar3)).EndInit();
            this.tabPage4.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize) (this.trackBar4)).EndInit();
            ((System.ComponentModel.ISupportInitialize) (this.numericUpDown1)).EndInit();
            this.ResumeLayout(false);
        }

        #endregion

        /// <summary>
        ///     The main entry point for the application.
        /// </summary>
        [STAThread]
        private static void Main()
        {
            Application.Run(new Form1());
        }

        #endregion
    }
}