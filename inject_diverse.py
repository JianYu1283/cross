"""Diversified cross-contract reentrancy benchmark (P2b).

Three distinct vulnerable patterns (each victim + reentering attacker), plus
clean-but-similar negatives (safe external-call patterns), so that the
cross-contract evaluation can report precision, not recall alone.

Templates (all reentrancy-positive; the victim carries the bug):
  T1  classic call.value    -- external call BEFORE balance write
  T2  delegatecall          -- delegatecall to a target BEFORE state write
  T3  cross-function (DAO)  -- call.value before credit write + stale-read sibling fn

Clean negatives (look like the above but are safe):
  C1  checks-effects-interactions -- state write BEFORE the external call
  C2  mutex-guarded                -- call.value with a reentrancy lock
"""
import os
import sys

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "injected_diverse"))


def T1_victim(n):
    return (
        f"contract Victim{n} {{\n"
        "    mapping(address => uint256) public balances;\n"
        "    function deposit() public payable { balances[msg.sender] += msg.value; }\n"
        "    function withdraw(uint256 amount) public {\n"
        "        require(balances[msg.sender] >= amount);\n"
        "        msg.sender.call.value(amount)();\n"
        "        balances[msg.sender] -= amount;\n"
        "    }\n"
        "}\n"
    )


def T2_victim(n):
    return (
        f"contract Victim{n} {{\n"
        "    mapping(address => uint256) public balances;\n"
        "    address public impl;\n"
        "    function deposit() public payable { balances[msg.sender] += msg.value; }\n"
        "    function setImpl(address _i) public { impl = _i; }\n"
        "    function execute() public {\n"
        "        require(balances[msg.sender] > 0);\n"
        "        impl.delegatecall(msg.data);\n"
        "        balances[msg.sender] = 0;\n"
        "    }\n"
        "}\n"
    )


def T3_victim(n):
    return (
        f"contract Victim{n} {{\n"
        "    mapping(address => uint256) public credit;\n"
        "    function donate(address to) public payable { credit[to] += msg.value; }\n"
        "    function withdraw(uint256 amount) public {\n"
        "        if (credit[msg.sender] >= amount) {\n"
        "            msg.sender.call.value(amount)();\n"
        "            credit[msg.sender] -= amount;\n"
        "        }\n"
        "    }\n"
        "    function transferTo(address to, uint256 amount) public {\n"
        "        require(credit[msg.sender] >= amount);\n"
        "        credit[to] += amount;\n"
        "        credit[msg.sender] -= amount;\n"
        "    }\n"
        "}\n"
    )


def attacker(n):
    return (
        f"contract Attacker{n} {{\n"
        f"    Victim{n} public victim;\n"
        "    uint256 public count;\n"
        f"    function Attacker{n}(Victim{n} _v) public {{ victim = _v; }}\n"
        "    function attack() public payable { victim.withdraw(1); }\n"
        "    function () public payable {\n"
        "        if (count < 5) { count += 1; victim.withdraw(1); }\n"
        "    }\n"
        "}\n"
    )


def C1_safe(n):
    return (
        f"contract Safe{n} {{\n"
        "    mapping(address => uint256) public balances;\n"
        "    function deposit() public payable { balances[msg.sender] += msg.value; }\n"
        "    function withdraw(uint256 amount) public {\n"
        "        require(balances[msg.sender] >= amount);\n"
        "        balances[msg.sender] -= amount;\n"
        "        msg.sender.call.value(amount)();\n"
        "    }\n"
        "}\n"
    )


def C2_safe(n):
    return (
        f"contract Safe{n} {{\n"
        "    mapping(address => uint256) public balances;\n"
        "    bool private locked;\n"
        "    function deposit() public payable { balances[msg.sender] += msg.value; }\n"
        "    function withdraw(uint256 amount) public {\n"
        "        require(!locked);\n"
        "        locked = true;\n"
        "        require(balances[msg.sender] >= amount);\n"
        "        msg.sender.call.value(amount)();\n"
        "        balances[msg.sender] -= amount;\n"
        "        locked = false;\n"
        "    }\n"
        "}\n"
    )


def main():
    n_pos_per_template = 13
    n_neg_per_type = 10
    pdir = os.path.join(OUT, "positive")
    ndir = os.path.join(OUT, "negative")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(ndir, exist_ok=True)

    templates = [("t1", T1_victim), ("t2", T2_victim), ("t3", T3_victim)]
    idx = 0
    for tag, fn in templates:
        for k in range(n_pos_per_template):
            v = fn(idx)
            with open(os.path.join(pdir, f"{tag}_victim_{idx}.sol"), "w") as f:
                f.write("pragma solidity ^0.4.24;\n" + v)
            # attacker only for t1/t3 (call.value reentry); t2 uses delegatecall target
            if tag in ("t1", "t3"):
                a = attacker(idx)
                with open(os.path.join(pdir, f"{tag}_pair_{idx}.sol"), "w") as f:
                    f.write("pragma solidity ^0.4.24;\n" + v + "\n" + a)
            idx += 1

    for k in range(n_neg_per_type):
        with open(os.path.join(ndir, f"c1_safe_{k}.sol"), "w") as f:
            f.write("pragma solidity ^0.4.24;\n" + C1_safe(k))
    for k in range(n_neg_per_type):
        with open(os.path.join(ndir, f"c2_safe_{k}.sol"), "w") as f:
            f.write("pragma solidity ^0.4.24;\n" + C2_safe(k))

    print(f"generated {idx} positive victims (+ t1/t3 pairs) and {2*n_neg_per_type} negatives in {OUT}")


if __name__ == "__main__":
    main()
