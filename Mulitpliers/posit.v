`timescale 1ns / 1ps
`define N 10
`define es 4
`define regime 2
module posit(a,psign,pexp,pfrac,pzero,pinf);
    input [`N-1:0]a;
    output psign;
    output [`es-1:0]pexp;
    output [`N-`es-`regime-2:0]pfrac;
    output pzero;
    output pinf;
    wire [`N-1:0]x;
//-----------------------sign----------------------
    assign psign = a[`N-1];
    assign x = psign? ~a[`N-2:0] + 1'b1: a[`N-2:0];

//-----------------------inf\zero check----------------------
    assign pzero = psign ? 1'b0:( x[`N-2:0]==`N-1'b0 ? 1'b1:1'b0);
    assign pinf = psign ? ( x[`N-2:0]==`N-1'b0 ? 1'b1:1'b0):1'b0;



//-----------------------expo----------------------
    assign pexp = x[`N-`regime-2:`N-`regime-`es-1];
//-----------------------frac-----------------------
    assign pfrac = x[`N-`regime-`es-2:0];

endmodule
//---------posit multiplication------------------------
module pmult(a,b,out,clk,reset,pinf,pzero);
    input [`N-1:0]a,b;
    input clk,reset;
    output reg [`N-1:0]out;
    output pinf,pzero;
    reg [`N-1:0]x,y;
    wire [`N-1:0]OUT;
    wire [`N-1:0]result;
    wire [2*(`N-`es-`regime)-1:0]mult_out;
    reg [`regime-1:0]result_out;
    wire psign;
    wire [`regime-1:0]pregime;
    wire [`regime+`es-1:0]a_exp,b_exp,t_exp;
    wire [`N-`es-`regime-2:0]pfrac;
    wire a_psign,b_psign;
    wire [`es-1:0]a_pexp,b_pexp;
    wire [`N-`es-`regime-2:0]a_pfrac,b_pfrac;
    wire a_pzero,a_pinf,b_pzero,b_pinf;

always@(posedge clk) begin
if(reset) begin
    x <= `N'b0;
    y <= `N'b0;
    out <= `N'b0;	end
else begin
    x <= a;
    y <= b;
    out <= OUT;	end
end
    posit a1(x,a_psign,a_pexp,a_pfrac,a_pzero,a_pinf);
    posit b1(y,b_psign,b_pexp,b_pfrac,b_pzero,b_pinf);
    assign pinf = (a_pinf | b_pinf) ? 1'b1 : 1'b0;
    assign pzero = ~(a_pinf|b_pinf)&(a_pzero|b_pzero) ? 1'b1 : 1'b0;

//-----------------------sign----------------------
    assign psign = a_psign ^ b_psign;

//-----------------------frac----------------------
    assign result[`N-`es-`regime-2:0] = mult_out[2*(`N-`es-`regime)-1]? mult_out[2*(`N-`es-`regime)-1-1:2*(`N-`es-`regime)-1-(`N-`es-`regime-1)] : mult_out[2*(`N-`es-`regime)-1-1-1:2*(`N-`es-`regime)-1-(`N-`es-`regime-1)-1];

//-----------------------total_exp----------------------
    assign a_exp =  a_pexp;
    assign b_exp =  b_pexp;
    assign t_exp = a_exp + b_exp + mult_out[2*(`N-`es-`regime)-1];
    assign pregime = t_exp[`es+`regime-1:`es];

    assign result[`N-`regime-2:`N-`es-`regime-1] = t_exp[`es-1:0];

always@(*) begin
case(pregime)
    2'b11 : result_out = 2'b01;
    2'b10 : result_out = 2'b00;
    2'b00 : result_out = 2'b10;
    default : result_out = 2'b11;
endcase
end

    assign result[`N-1] = psign;
    assign result[`N-2:`N-`regime-1] = result_out;
    assign OUT = result[`N-1]? {result[`N-1],~result[`N-2:0]+1'b1} : result;
endmodule