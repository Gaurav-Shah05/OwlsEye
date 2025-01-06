nt main() {
    int nsize = 20;
    int es = 2;
    int rf = 4;
    float scale = 1.0;

    uint32_t int32_constants[16];
    uint64_t int64_constants[2];

    generate_fixedposit_constants(nsize, es, rf, int32_constants, int64_constants);

    for (float temp_input = -0.4f; temp_input <= 0.1f; temp_input += 0.000255f) {
        // ufloat tempf;
        // temp_input = temp_input*scale;
        // tempf.f = temp_input;

        uint32_t temp = fp32tofixedp16(temp_input*scale, int32_constants, int64_constants);
        float output = fixedp16tofp32(temp, int32_constants, int64_constants);
        output = output/scale;
        printf("input: %f output: %f\n", temp_input, output);
        // temp_input = temp_input/scale;
    }

    return 0;
}