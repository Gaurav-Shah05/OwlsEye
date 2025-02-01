`timescale 1ns / 1ps
`define N 8
`define es 2
`define regime 2
module posit(a,psign,pexp,pfrac,pzero,pinf);
    input [`N-1:0]a;
    output psign;
    //output [`regime-1:0]pregime;
    output [`es-1:0]pexp;
    output [`N-`es-`regime-2:0]pfrac;
    output pzero;
    output pinf;
    wire [`N-2:0]x;
    //wire [`regime-1:0]number;
    //reg [`regime-1:0]p_regime;

//-----------------------sign----------------------
    assign psign = a[`N-1];
    assign x = psign? ~a[`N-2:0] + 1'b1: a[`N-2:0];

//-----------------------inf\zero check----------------------
    assign pzero = psign ? 1'b0:( x[`N-2:0]==`N-1'b0 ? 1'b1:1'b0);
    assign pinf = psign ? ( x[`N-2:0]==`N-1'b0 ? 1'b1:1'b0):1'b0;

//-----------------------regime----------------------/
    /* assign number = x[`N-2]?~x[`N-2:`N-2-`regime+1]:x[`N-2:`N-2-`regime+1];
	always@(*) begin
	case(number)
   	 2'b01 : p_regime = 2'b11;
    	default : p_regime = 2'b10;
endcase
end

    assign pregime = x[`N-2]?p_regime-1'b1:~p_regime+1'b1;*/
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
    reg [`N-1:0]OUT;
    wire [`N-1:0]result;
    wire [2*(`N-`es-`regime)-1:0]mult_out;
    reg [`regime-1:0]result_out;
    reg [`es-1:0]t_out;
    wire psign;
    wire uo;
    wire pregime;
    wire [`es:0]a_exp,b_exp,t_exp;
    wire [`N-`es-`regime-2:0]pfrac;
    wire a_psign,b_psign;
    //wire [`regime-1:0]a_pregime,b_pregime;
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
    out <= OUT;
	end
end
    posit a1(x,a_psign,a_pexp,a_pfrac,a_pzero,a_pinf);
    posit b1(y,b_psign,b_pexp,b_pfrac,b_pzero,b_pinf);
    assign pinf = (a_pinf | b_pinf) ? 1'b1 : 1'b0;
    assign pzero = ~(a_pinf|b_pinf)&(a_pzero|b_pzero) ? 1'b1 : 1'b0;

//-----------------------sign----------------------
    assign psign = a_psign ^ b_psign;

//-----------------------frac----------------------
    assign mult_out={1'b1,a_pfrac}*{1'b1,b_pfrac};
    assign result[`N-`es-`regime-2:0] = mult_out[2*(`N-`es-`regime)-1]? mult_out[2*(`N-`es-`regime)-1-1:2*(`N-`es-`regime)-1-(`N-`es-`regime-1)] : mult_out[2*(`N-`es-`regime)-1-1-1:2*(`N-`es-`regime)-1-(`N-`es-`regime-1)-1];

//-----------------------total_exp----------------------
    //assign a_exp = {a_pregime,`es'b0} + a_pexp;
    //assign b_exp = {b_pregime,`es'b0} + b_pexp;
    assign t_exp = a_pexp + b_pexp + mult_out[2*(`N-`es-`regime)-1];
    assign pregime = t_exp[`es];

    assign result[`N-`regime-2:`N-`es-`regime-1] = t_exp[`es-1:0];
    assign uo = pregime?1'b0:1'b1;

    assign result[`N-`regime-2:`N-`es-`regime-1] = t_exp[`es-1:0];
    assign result[`N-1] = psign;
    assign result[`N-2:`N-`regime-1] = 2'b01;

    always@(*) begin
        case(uo) 
	    1'b1 : OUT = psign ? 8'b11100000 : 8'b00100000;
        1'b0: OUT = (result[`N-1]? {result[`N-1],~result[`N-2:0]+1'b1} : result);
        endcase
    end
endmodule