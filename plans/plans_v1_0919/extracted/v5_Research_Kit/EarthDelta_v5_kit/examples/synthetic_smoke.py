"""Plumbing smoke test only: random projected arrays are NOT ERA5 forecasts."""
import json
import torch
from earthdelta_v5 import PairedEditPredictor, pilot_plans, make_training_targets, select_plan

def main():
    torch.manual_seed(42);torch.set_num_threads(1)
    plans=pilot_plans();enabled=torch.tensor([any(p.groups) for p in plans])
    descriptors=torch.tensor([p.descriptor() for p in plans])
    net=PairedEditPredictor(6,4,9,6,16)
    history=torch.randn(8,4,3,6);memory=torch.randn(8,3,4);leads=torch.tensor([6.,12.,24.])
    ref=torch.randn(8,3,3,6);edited=ref[:,None]+0.1*torch.randn(8,6,3,3,6);edited[:,0]=ref
    targets=make_training_targets(ref+torch.randn_like(ref),ref,edited)
    optim=torch.optim.Adam(net.parameters(),lr=1e-3)
    first=last=None
    for step in range(30):
        optim.zero_grad();out=net(history,memory,descriptors,leads,enabled)
        loss=net.training_loss(out,targets,enabled)['total']
        if step==0:first=loss.item()
        loss.backward();optim.step();net.update_targets();last=loss.item()
    with torch.no_grad():
        gains=net(history,memory,descriptors,leads,enabled)['gain'].mean((-1,-2))
        # These numbers are toy cost units, not measured weather/GPU milliseconds.
        chosen=select_plan(gains,torch.tensor([1.,2.,2.,3.,3.,5.]),3.,tuple(p.plan_id for p in plans))
    print(json.dumps({'status':'synthetic_plumbing_only_not_weather_evidence','train_steps':30,
                      'first_training_loss':first,'last_training_loss':last,
                      'selected_plans':[plans[i].plan_id for i in chosen.tolist()]},indent=2))

if __name__=='__main__':main()
