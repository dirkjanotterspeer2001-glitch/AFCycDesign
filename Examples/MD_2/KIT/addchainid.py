input_pdb = "cluster_03_KIT_without_water.pdb"
output_pdb = "cluster_03_KIT_without_water_update.pdb"

with open(input_pdb) as f_in, open(output_pdb, "w") as f_out:

    for line in f_in:

        if line.startswith(("ATOM", "HETATM")):

            resname = line[17:21].strip()

            if len(resname) == 4 and resname.startswith("C"):
                new_resname = resname[1:]

            elif len(resname) == 4 and resname.startswith("N"):
                new_resname = resname[1:]

            # CYX -> CYS
            elif resname == "CYX":
                new_resname = "CYS"

            elif resname == "HID":
                new_resname = "HIS"

            else:
                new_resname = resname

            # FIXED: schrijf exact 4-char veld (PDB kolom 17–20)
            line = line[:17] + f"{new_resname:<4}" + line[21:]

            # chain ID fix
            if line[21] == " ":
                line = line[:21] + "A" + line[22:]

        f_out.write(line)

print("Saved:", output_pdb)
