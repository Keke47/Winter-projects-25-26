`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02.07.2026 04:34:59
// Design Name: 
// Module Name: output_layer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module output_layer(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire signed [15:0] data_in,
    input  wire signed [15:0] weight_n1,
    input  wire signed [15:0] weight_n2,
    input  wire signed [15:0] weight_n3,
    input  wire [15:0] bias_1, bias_2, bias_3,
    input  wire        last,
    output wire [15:0] out_1, out_2, out_3,
    output wire        valid_1, valid_2, valid_3
    );
 
    neuron neuron1(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n1), .bias(bias_1), .last(last), .out(out_1), .valid(valid_1));
    neuron neuron2(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n2), .bias(bias_2), .last(last), .out(out_2), .valid(valid_2));
    neuron neuron3(.clk(clk), .rst_n(rst_n), .start(start), .data_in(data_in), .weight_in(weight_n3), .bias(bias_3), .last(last), .out(out_3), .valid(valid_3));
 
endmodule
