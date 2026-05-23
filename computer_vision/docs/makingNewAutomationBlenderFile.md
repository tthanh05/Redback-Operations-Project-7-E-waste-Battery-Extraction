## 📦 Blender Automation Guide (Synthetic Data Generation)

This guide explains how to create a new Blender file to generate synthetic images for different phone models.

🔗 Blender Files & Assets:
https://deakin365-my.sharepoint.com/:f:/g/personal/s224432628_deakin_edu_au/IgBLfA01_4KVRZVr7TVt6VMXAUx3ADw2ayMuL0OPEtSbbGs?e=iPzT3q

---

### 🛠️ Steps

1. **Duplicate an Existing File**

   * Copy an existing `.blend` file (e.g., `s25.blend`)

2. **Load Script**

   * Open Blender → Go to **Scripting Tab**
   * Ensure `blenderGenerationScript.py` is loaded
   * If not, paste script manually

3. **Replace Phone Model**

   * Go to **Layout Tab**
   * Locate object named (e.g., `s25`)
   * This represents current phone internals

4. **Import New Phone Internals**

   * Click: `Add → Image → Mesh Plane`
   * Select image from `phoneInsides` folder

5. **Move to Correct Collection**

   * Place mesh inside correct collection group
   * Expand object tree if needed

6. **Delete Old Model**

   * Remove previous phone internals (e.g., `s25`)

7. **Transform New Model**

   * Position:

     * Z-axis = **0.29 m**
   * Rotation:

     * Adjust Z-axis only
   * Scale:

     * Fit mesh to phone frame

8. **Generate Images**

   * Go to **Scripting Tab**
   * Modify `while loop` image count
   * Click ▶️ Run Script

---

### 📌 Notes

* Ensure aspect ratio alignment between mesh and phone case
* Adjust rotation carefully for correct orientation
* Output images will be used for YOLO training

---
