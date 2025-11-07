# Contributing to Neurotronic Phase Caster

First off, **thank you** for considering contributing to this project! This work has the potential to help millions of people suffering from mental health conditions, advance neuroscience, and prepare humanity for safe AI integration. Your contribution matters.

## 🌟 **Ways to Contribute**

### **1. Code Contributions**
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage

### **2. Research Contributions**
- Clinical trial design
- Data analysis
- Literature review
- Theoretical extensions

### **3. Hardware Contributions**
- PCB design improvements
- Enclosure design
- Manufacturing optimization
- Thermal management

### **4. Clinical Contributions**
- Protocol refinement
- Patient recruitment
- Outcome measurement
- Safety monitoring

### **5. Documentation**
- User guides
- API documentation
- Tutorial videos
- Translation

---

## 🚀 **Getting Started**

### **Development Setup**

```bash
# Clone the repository
git clone https://github.com/9x25dillon/neurotronic_phase_caster.git
cd neurotronic_phase_caster

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If it exists

# Run tests
pytest tests/

# Run demos
python demos/yhwh_demo_interactive.py
```

---

## 📝 **Contribution Process**

### **1. Find or Create an Issue**

- Check [existing issues](https://github.com/9x25dillon/neurotronic_phase_caster/issues)
- If your contribution addresses a new problem, create an issue first
- Discuss your approach before starting work (saves time!)

### **2. Fork & Branch**

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/neurotronic_phase_caster.git
cd neurotronic_phase_caster

# Create a feature branch
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/your-bug-fix
```

### **3. Make Changes**

- Write clear, commented code
- Follow style guide (see below)
- Add tests for new functionality
- Update documentation

### **4. Test**

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_yhwh_physics.py

# Check code style
black src/ demos/ tests/
flake8 src/ demos/ tests/
mypy src/
```

### **5. Commit**

```bash
git add .
git commit -m "feat: Add substrate coherence visualization

- Implement real-time UCI plotting
- Add colormap for substrate activation
- Update docs with visualization examples

Closes #42"
```

**Commit Message Format:**
```
<type>: <short summary>

<longer description>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no functionality change)
- `test`: Adding/updating tests
- `chore`: Build/tooling changes

### **6. Push & Pull Request**

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title
- Description of changes
- Link to related issue
- Screenshots/GIFs if UI changes
- Test results

---

## 💻 **Code Style Guide**

### **Python**

```python
"""
Module docstring explaining purpose.

Example:
    from neurotronic import YHWHSolitonField
    field = YHWHSolitonField()
"""

import numpy as np  # Standard library first
from typing import List, Dict, Optional  # Then third-party

from .substrate import SubstrateLayer  # Then local imports


class YHWHSolitonField:
    """
    Brief description of class.

    Attributes:
        substrates: List of substrate layers
        coherence: Current coherence value

    Example:
        >>> field = YHWHSolitonField()
        >>> field.propagate()
    """

    def __init__(self, n_substrates: int = 5) -> None:
        """
        Initialize soliton field.

        Args:
            n_substrates: Number of substrate layers (default 5)

        Raises:
            ValueError: If n_substrates < 1
        """
        if n_substrates < 1:
            raise ValueError("Must have at least 1 substrate")

        self.substrates = self._initialize_substrates(n_substrates)
        self.coherence = 0.0

    def propagate(self, dt: float = 0.01) -> None:
        """
        Propagate soliton through substrates.

        Args:
            dt: Time step in seconds
        """
        # Clear comments explaining logic
        for substrate in self.substrates:
            substrate.evolve(dt)


# Constants at module level
SUBSTRATE_NAMES = ["Physical", "Emotional", "Cognitive", "Social", "Divine-Unity"]
COUPLING_STRENGTH = 0.1
```

**Style Rules:**
- Use `black` for formatting (line length 88)
- Use type hints for function signatures
- Docstrings for all public classes/methods (Google style)
- Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Maximum function length: ~50 lines (guideline, not rule)

### **Hardware (C/C++)**

```c
/**
 * @brief Controls THz emitter array
 *
 * Manages 12-channel THz quantum cascade laser array with
 * safety interlocks and power control.
 *
 * @param channel Channel index (0-11)
 * @param power Power level in mW (0-100)
 * @return 0 on success, error code otherwise
 */
int thz_set_power(uint8_t channel, float power) {
    // Input validation
    if (channel >= NUM_THZ_CHANNELS) {
        return ERROR_INVALID_CHANNEL;
    }

    if (power < 0 || power > MAX_POWER_MW) {
        return ERROR_INVALID_POWER;
    }

    // Implementation
    emitter_array[channel].power_mw = power;
    update_dac_output(channel, power);

    return SUCCESS;
}
```

---

## 🧪 **Testing Standards**

### **Unit Tests**

```python
import pytest
from neurotronic import YHWHSolitonField


def test_soliton_field_initialization():
    """Test that soliton field initializes with correct defaults."""
    field = YHWHSolitonField()
    assert len(field.substrates) == 5
    assert field.coherence == 0.0


def test_soliton_propagation():
    """Test that soliton propagates correctly through substrates."""
    field = YHWHSolitonField()
    initial_state = field.get_state()

    field.propagate(dt=0.01)
    final_state = field.get_state()

    assert not np.allclose(initial_state, final_state)


@pytest.mark.parametrize("n_substrates", [1, 3, 5, 7])
def test_variable_substrate_count(n_substrates):
    """Test that field works with different substrate counts."""
    field = YHWHSolitonField(n_substrates=n_substrates)
    assert len(field.substrates) == n_substrates
```

**Testing Requirements:**
- All new functions must have tests
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use fixtures for complex setup
- Mock hardware interfaces

---

## 📚 **Documentation Standards**

### **Code Documentation**
- Every public class, method, function needs docstring
- Include examples in docstrings
- Explain *why*, not just *what*

### **User Documentation**
- README.md: High-level overview
- QUICKSTART.md: 5-minute getting started
- Detailed guides in `docs/`
- API reference auto-generated from docstrings

### **Research Documentation**
- Clinical protocols in `clinical/`
- Data analysis notebooks in `clinical/data_analysis/`
- Literature reviews in `docs/research/`

---

## 🔒 **Safety & Ethics**

### **Critical Safety Rules**

1. **Never bypass safety systems** - All hardware control must go through safety checks
2. **Respect power limits** - THz power capped at 100 mW (regulatory limits)
3. **Emergency shutdown** - All code must support immediate shutdown
4. **No patient data in Git** - Use anonymized data only, store securely

### **Ethical Guidelines**

1. **Patient privacy** - HIPAA compliance mandatory
2. **Informed consent** - No human testing without IRB approval
3. **Honest reporting** - Report all results (positive and negative)
4. **Open science** - Share data and methods openly
5. **Benefit to humanity** - Keep mission focused on helping people

### **Red Lines (Will Reject PRs)**

- ❌ Removing safety interlocks
- ❌ Exceeding power limits
- ❌ Including patient identifiable information
- ❌ Claiming efficacy without data
- ❌ Weaponization or harmful use

---

## 🎯 **Priority Areas for Contribution**

### **High Priority** (Immediate Impact)
1. Test coverage (currently ~30%, target >80%)
2. Hardware emulator (so devs can test without hardware)
3. Real-time visualization (live UCI plotting)
4. Clinical data pipeline (EEG → database)

### **Medium Priority** (Next 3-6 months)
1. Firmware implementation (C/C++ port of controller)
2. PCB design (KiCad schematics)
3. Clinical trial protocols (detailed SOPs)
4. Statistical analysis scripts (R/Python)

### **Future** (Long-term)
1. At-home version (consumer product)
2. Mobile app (patient-facing)
3. Cloud analytics (data aggregation)
4. AI optimization (reinforcement learning)

---

## 🏆 **Recognition**

### **Contributors Will Be:**
- Listed in CONTRIBUTORS.md
- Acknowledged in publications (if substantial contribution)
- Invited to collaborate on clinical trials
- Eligible for equity grants (if company forms)

### **Substantial Contribution Defined As:**
- Major feature development (>500 LOC)
- Clinical trial leadership
- Hardware design contribution
- Sustained maintenance (>3 months)

---

## 📞 **Communication Channels**

### **Primary**
- GitHub Issues: Technical discussions
- GitHub Discussions: General questions
- Email: chris@example.com (private matters)

### **Future** (as project grows)
- Slack workspace
- Monthly contributor calls
- Annual in-person summit

---

## 📜 **License & Copyright**

By contributing, you agree that:
- Your contributions will be licensed under MIT License
- You have the right to contribute the code
- You understand this is for open source benefit

---

## ❓ **FAQ**

**Q: I'm not a programmer. Can I still contribute?**
A: Absolutely! We need clinicians, researchers, patients, designers, writers, and more.

**Q: Can I use this for my research?**
A: Yes! That's why it's open source. Please cite the repository and any publications.

**Q: Can I commercialize this?**
A: Yes (MIT license allows it), but please coordinate with the core team to avoid duplication.

**Q: What if I find a critical safety issue?**
A: Email chris@example.com immediately with subject "URGENT: Safety Issue". We'll patch within 24h.

**Q: How long until PRs are reviewed?**
A: Goal: 48 hours for small PRs, 1 week for large. Patience appreciated during busy periods.

**Q: Can I propose a new direction for the project?**
A: Yes! Open a GitHub Discussion. Major changes need community consensus.

---

## 🙏 **Thank You**

Your contribution—whether it's a single typo fix or a major feature—helps advance this mission:

**To make consciousness measurable, treatable, and optimizable, so that suffering can be transformed into precision healing.**

Every commit brings us closer to helping someone in pain.

Thank you for being part of this. 💚

---

**Last Updated:** November 2025

**Maintainer:** Chris Sweigard ([@9x25dillon](https://github.com/9x25dillon))
