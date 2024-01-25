import openvino.runtime as ov

core = ov.Core()
model = core.read_model("/home/vshampor/work/optimum-intel/ov_model/openvino_model.xml")

if model.has_rt_info("gguf_params"):
    print("VSHAMPOR: model has rt_info")
    gguf_params = model.get_rt_info("gguf_params")
    gguf_params_dict = gguf_params.astype(dict)
    print(list(gguf_params_dict.keys()))
    tensor_map = gguf_params_dict["tensor_name_map"]
    tensor_shape_map = gguf_params_dict["tensor_shape_map"]
    tensor_map_decoded = {k: v.astype(str) for k, v in tensor_map.items()}
    tensor_shape_map_decoded = {k: v.astype(str).replace(' ', '') for k, v in tensor_shape_map.items()}
    print(f"VSHAMPOR: {len(tensor_map_decoded)} tensors required by GGUF")
    const_op_nodes = [o for o in model.get_ops() if o.get_type_name() == "Constant"]
    print(f"VSHAMPOR: {len(const_op_nodes)} const op nodes in model")
    const_op_node_names = list(o.get_friendly_name() for o in const_op_nodes)
    matches_dict = {}
    for torch_name in tensor_map_decoded:
        this_tensor_exact_matches = []
        this_tensor_partial_matches = []
        expected_shape = tensor_shape_map_decoded[torch_name]
        for co_node in const_op_nodes:
            co_name = co_node.get_friendly_name()
            if torch_name in co_name:
                shape_str = str(co_node.get_output_shape(0))
                if shape_str == expected_shape:
                    this_tensor_exact_matches.append(co_name)
                else:
                    print(f"shape mismatch: expected {expected_shape} vs actual {shape_str} for {co_name}")
        if not this_tensor_exact_matches:
            prefix = torch_name.split('.weight')[0].split('.bias')[0]
            for co_node in const_op_nodes:
                co_name = co_node.get_friendly_name()
                shape_str = str(co_node.get_output_shape(0))
                if prefix in co_name:
                    if shape_str == expected_shape:
                        this_tensor_partial_matches.append(co_name)
                    else:
                        print(f"shape mismatch: expected {expected_shape} vs actual {shape_str} for {co_name}")
        matches_dict[torch_name] = (this_tensor_exact_matches, this_tensor_partial_matches)

    exact_matched_tensors = 0
    multiple_exact_matched_tensors = 0
    partial_matched_tensors = 0
    multiple_partial_matched_tensors = 0
    well_posed_matches = 0
    no_matches = []
    for torch_name, match_list_tuple in matches_dict.items():
        exact_matches, partial_matches = match_list_tuple
        if exact_matches:
            print(f"{torch_name}: {len(exact_matches)} exact matches -> {exact_matches}")
            exact_matched_tensors += 1
            if len(exact_matches) > 1:
                multiple_exact_matched_tensors += 1
            else:
                well_posed_matches += 1
        elif partial_matches:
            print(f"{torch_name}: {len(partial_matches)} partial matches -> {partial_matches}")
            partial_matched_tensors += 1
            if len(partial_matches) > 1:
                multiple_partial_matched_tensors += 1
            else:
                well_posed_matches += 1
        else:
            print(f"{torch_name}: no matches")
            no_matches.append(torch_name)

    total_tensor_count = len(tensor_map)
    print(f"EXACT MATCHES: {exact_matched_tensors}/{total_tensor_count} ({multiple_exact_matched_tensors} multiple)")
    print(f"PARTIAL MATCHES: {partial_matched_tensors}/{total_tensor_count} ({multiple_partial_matched_tensors} multiple)")
    print(f"NO MATCHES FOR: {no_matches}")
    print(f"WELL-POSED MATCHES: {well_posed_matches}/{total_tensor_count}")

else:
    print("VSHAMPOR: model has no rt_info")

