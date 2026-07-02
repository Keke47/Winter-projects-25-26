`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 02.07.2026 04:36:36
// Design Name: 
// Module Name: mlptop
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


module mlptop #(
    parameter WEIGHT_FILE = "weights.mem",
    parameter BIAS_FILE   = "biases.mem"
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire                start,      // pulse 1 cycle to kick off one inference
    input  wire signed [15:0]  x1, x2, x3, x4,  // network inputs, Q8.8 fixed point
    output reg  [15:0]         y1, y2, y3,  // output-layer results (post-ReLU logits)
    output reg                 done         // pulses high for 1 cycle when y1..y3 valid
);
    reg signed [15:0] weight_mem [0:55];
    reg        [15:0] bias_mem   [0:10];
    initial begin
        $readmemb(WEIGHT_FILE, weight_mem);
        $readmemb(BIAS_FILE,   bias_mem);
    end
    reg                 h_start, h_last;
    reg  signed [15:0]  h_data_in;
    reg  signed [15:0]  h_w1, h_w2, h_w3, h_w4, h_w5, h_w6, h_w7, h_w8;
    wire        [15:0]  h_out1, h_out2, h_out3, h_out4, h_out5, h_out6, h_out7, h_out8;
    wire                h_valid1, h_valid2, h_valid3, h_valid4, h_valid5, h_valid6, h_valid7, h_valid8;
 
    layer hidden_layer (
        .clk(clk), .rst_n(rst_n),
        .start(h_start),
        .data_in(h_data_in),
        .weight_n1(h_w1), .weight_n2(h_w2), .weight_n3(h_w3), .weight_n4(h_w4),
        .weight_n5(h_w5), .weight_n6(h_w6), .weight_n7(h_w7), .weight_n8(h_w8),
        .bias_1(bias_mem[0]), .bias_2(bias_mem[1]), .bias_3(bias_mem[2]), .bias_4(bias_mem[3]),
        .bias_5(bias_mem[4]), .bias_6(bias_mem[5]), .bias_7(bias_mem[6]), .bias_8(bias_mem[7]),
        .last(h_last),
        .out_1(h_out1), .out_2(h_out2), .out_3(h_out3), .out_4(h_out4),
        .out_5(h_out5), .out_6(h_out6), .out_7(h_out7), .out_8(h_out8),
        .valid_1(h_valid1), .valid_2(h_valid2), .valid_3(h_valid3), .valid_4(h_valid4),
        .valid_5(h_valid5), .valid_6(h_valid6), .valid_7(h_valid7), .valid_8(h_valid8)
    );
 
    reg                 o_start, o_last;
    reg  signed [15:0]  o_data_in;
    reg  signed [15:0]  o_w1, o_w2, o_w3;
    wire        [15:0]  o_out1, o_out2, o_out3;
    wire                o_valid1, o_valid2, o_valid3;
 
    output_layer out_layer (
        .clk(clk), .rst_n(rst_n),
        .start(o_start),
        .data_in(o_data_in),
        .weight_n1(o_w1), .weight_n2(o_w2), .weight_n3(o_w3),
        .bias_1(bias_mem[8]), .bias_2(bias_mem[9]), .bias_3(bias_mem[10]),
        .last(o_last),
        .out_1(o_out1), .out_2(o_out2), .out_3(o_out3),
        .valid_1(o_valid1), .valid_2(o_valid2), .valid_3(o_valid3)
    );
 reg [15:0] h_act [0:7];
 localparam S_IDLE    = 4'd0,
               S_H_PRIME  = 4'd1,   // start=1: clear hidden-layer accumulators
               S_H_FEED   = 4'd2,   // stream x1..x4, one per cycle
               S_H_WAIT   = 4'd3,   // 1 bubble cycle for the registered valid/out
               S_H_LATCH  = 4'd4,   // capture the 8 hidden outputs
               S_O_PRIME  = 4'd5,   // start=1: clear output-layer accumulators
               S_O_FEED   = 4'd6,   // stream h_act[0..7], one per cycle
               S_O_WAIT   = 4'd7,   // 1 bubble cycle
               S_O_LATCH  = 4'd8,   // capture y1..y3, raise done
               S_DONE     = 4'd9;
 
    reg [3:0] state;
    reg [1:0] h_idx;   // 0..3
    reg [2:0] o_idx;   // 0..7
 
    always @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            h_start   <= 1'b0; h_last <= 1'b0; h_data_in <= 16'd0;
            h_w1 <= 0; h_w2 <= 0; h_w3 <= 0; h_w4 <= 0;
            h_w5 <= 0; h_w6 <= 0; h_w7 <= 0; h_w8 <= 0;
            o_start   <= 1'b0; o_last <= 1'b0; o_data_in <= 16'd0;
            o_w1 <= 0; o_w2 <= 0; o_w3 <= 0;
            h_idx <= 2'd0; o_idx <= 3'd0;
            done  <= 1'b0;
            y1 <= 16'd0; y2 <= 16'd0; y3 <= 16'd0;
        end else begin
            // defaults: single-cycle pulses go low unless re-asserted below
            h_start <= 1'b0; h_last <= 1'b0;
            o_start <= 1'b0; o_last <= 1'b0;
            done    <= 1'b0;
 
            case (state)
                S_IDLE: begin
                    if (start) begin
                        h_idx <= 2'd0;
                        state <= S_H_PRIME;
                    end
                end
 
                // clear hidden-layer accumulators before streaming real data
                S_H_PRIME: begin
                    h_start <= 1'b1;
                    state   <= S_H_FEED;
                end
 
                // one input value (x1..x4) + its 8 weights per cycle
                S_H_FEED: begin
                    case (h_idx)
                        2'd0: h_data_in <= x1;
                        2'd1: h_data_in <= x2;
                        2'd2: h_data_in <= x3;
                        2'd3: h_data_in <= x4;
                    endcase
                    h_w1 <= weight_mem[h_idx*8 + 0];
                    h_w2 <= weight_mem[h_idx*8 + 1];
                    h_w3 <= weight_mem[h_idx*8 + 2];
                    h_w4 <= weight_mem[h_idx*8 + 3];
                    h_w5 <= weight_mem[h_idx*8 + 4];
                    h_w6 <= weight_mem[h_idx*8 + 5];
                    h_w7 <= weight_mem[h_idx*8 + 6];
                    h_w8 <= weight_mem[h_idx*8 + 7];
 
                    if (h_idx == 2'd3) begin
                        h_last <= 1'b1;      // this cycle carries x4 -> final MAC term
                        state  <= S_H_WAIT;
                    end else begin
                        h_idx <= h_idx + 1'b1;
                    end
                end
 
                S_H_WAIT: begin
                    state <= S_H_LATCH;
                end
 
                S_H_LATCH: begin
                    h_act[0] <= h_out1; h_act[1] <= h_out2;
                    h_act[2] <= h_out3; h_act[3] <= h_out4;
                    h_act[4] <= h_out5; h_act[5] <= h_out6;
                    h_act[6] <= h_out7; h_act[7] <= h_out8;
                    o_idx    <= 3'd0;
                    state    <= S_O_PRIME;
                end
 
                // clear output-layer accumulators before streaming h_act
                S_O_PRIME: begin
                    o_start <= 1'b1;
                    state   <= S_O_FEED;
                end
 
                // one hidden activation + its 3 weights per cycle
                S_O_FEED: begin
                    o_data_in <= h_act[o_idx];
                    o_w1 <= weight_mem[32 + o_idx*3 + 0];
                    o_w2 <= weight_mem[32 + o_idx*3 + 1];
                    o_w3 <= weight_mem[32 + o_idx*3 + 2];
 
                    if (o_idx == 3'd7) begin
                        o_last <= 1'b1;      // this cycle carries h8 -> final MAC term
                        state  <= S_O_WAIT;
                    end else begin
                        o_idx <= o_idx + 1'b1;
                    end
                end
 
                S_O_WAIT: begin
                    state <= S_O_LATCH;
                end
 
                S_O_LATCH: begin
                    y1    <= o_out1;
                    y2    <= o_out2;
                    y3    <= o_out3;
                    done  <= 1'b1;
                    state <= S_DONE;
                end
 
                S_DONE: begin
                    state <= S_IDLE;
                end
 
                default: state <= S_IDLE;
            endcase
        end
    end
 
endmodule
